from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import csv
import hashlib

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# Paths and environment
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

CHROMA_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "chroma_rg271"
COLLECTION_NAME = "rg271_agentic_chunks"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG_PATH = REPORTS_DIR / "api_audit_log.csv"


# ------------------------------------------------------------
# Load shared resources once at startup
# ------------------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

openai_client = OpenAI()


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------

app = FastAPI(
    title="Complaint Intelligence Platform",
    description=(
        "PII-safe complaint classification and RG 271 RAG assistant "
        "for financial complaints."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only — restrict origins before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------

class ComplaintRequest(BaseModel):
    complaint_text: str = Field(
        ...,
        min_length=20,
        description="Raw or redacted consumer complaint narrative."
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Optional customer or case identifier. Do not send real PII in demo."
    )


class RetrievedContextItem(BaseModel):
    final_rank: int
    distance: float
    query_name: str
    title: str
    rg_refs: str
    chunk_type: str
    text_preview: str


class ComplaintAnalysisResponse(BaseModel):
    received_at: str
    input_preview: str
    predicted_label: str
    classification_confidence: float
    retrieved_rg_refs: list[str]
    retrieved_context: list[RetrievedContextItem]
    human_review_required: bool
    handling_note: str


# ------------------------------------------------------------
# RAG prompt
# ------------------------------------------------------------

RAG_SYSTEM_PROMPT = """
You are a complaint intelligence assistant for a financial services setting.

You must use only the provided RG 271 context.
Do not provide legal advice.
Do not invent regulatory references.
If the context is insufficient, say that human review is required.
Frame recommendations as internal handling considerations, not final instructions.
Avoid definitive operational commands such as "cease collection activity" or
"stop enforcement immediately" unless the retrieved context explicitly supports
that exact requirement. Prefer cautious language such as "assess whether",
"consider whether", "review whether", and "subject to applicable legal and
operational requirements".

Write a short internal complaint handling note with:
1. Complaint context
2. Relevant RG 271 guidance
3. Suggested handling considerations
4. Human review flag

Keep the tone professional, concise, and suitable for an internal complaints team.
"""


# ------------------------------------------------------------
# Retrieval helpers
# ------------------------------------------------------------

def build_retrieval_queries(complaint_text: str) -> list[dict]:
    short_text = complaint_text[:1200]

    return [
        {
            "query_name": "complaint_definition",
            "query_text": f"""
            Regulatory guidance on what counts as a complaint, expression of dissatisfaction,
            complainant, disputed transaction, and complaint handling.

            Complaint:
            {short_text}
            """
        },
        {
            "query_name": "idr_process_response",
            "query_text": f"""
            Regulatory guidance on internal dispute resolution process, IDR response,
            complaint handling obligations, written response, reasons for decision,
            and financial firm responsibilities.

            Complaint:
            {short_text}
            """
        },
        {
            "query_name": "idr_timeframes",
            "query_text": f"""
            Regulatory guidance on maximum IDR timeframes, acknowledgement of complaint,
            response time limits, complaint resolution timeframe, AFCA referral,
            and delay in complaint handling.

            Complaint:
            {short_text}
            """
        },
        {
            "query_name": "dispute_context",
            "query_text": f"""
            Consumer complaint involving disputed charges, disputed debt, account issue,
            transaction dispute, collection activity, financial hardship, enforcement,
            or unresolved complaint handling.

            Complaint:
            {short_text}
            """
        }
    ]


def retrieve_rg271_context(
    complaint_text: str,
    n_results_per_query: int = 5,
    top_k_final: int = 5
) -> list[dict]:
    queries = build_retrieval_queries(complaint_text)
    all_results = []

    for item in queries:
        query_embedding = embedding_model.encode(
            item["query_text"],
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results_per_query
        )

        for i in range(len(results["ids"][0])):
            all_results.append({
                "query_name": item["query_name"],
                "rank_within_query": i + 1,
                "id": results["ids"][0][i],
                "distance": float(results["distances"][0][i]),
                "title": results["metadatas"][0][i].get("title", ""),
                "rg_refs": results["metadatas"][0][i].get("rg_refs", ""),
                "chunk_type": results["metadatas"][0][i].get("chunk_type", ""),
                "text": results["documents"][0][i]
            })

    # Deduplicate by Chroma id and keep best distance
    best_by_id = {}

    for row in all_results:
        chunk_id = row["id"]

        if chunk_id not in best_by_id or row["distance"] < best_by_id[chunk_id]["distance"]:
            best_by_id[chunk_id] = row
        else:
            existing_queries = set(best_by_id[chunk_id]["query_name"].split(", "))
            existing_queries.add(row["query_name"])
            best_by_id[chunk_id]["query_name"] = ", ".join(sorted(existing_queries))

    deduped = sorted(best_by_id.values(), key=lambda x: x["distance"])[:top_k_final]

    for i, row in enumerate(deduped):
        row["final_rank"] = i + 1

    return deduped


def build_rag_context(retrieved_rows: list[dict]) -> str:
    context_parts = []

    for row in retrieved_rows:
        context_parts.append(
            f"""
Title: {row["title"]}
RG refs: {row["rg_refs"]}
Chunk type: {row["chunk_type"]}
Text:
{row["text"]}
"""
        )

    return "\n\n---\n\n".join(context_parts)


def generate_handling_note(complaint_text: str, retrieved_rows: list[dict]) -> str:
    rag_context = build_rag_context(retrieved_rows)

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": RAG_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Complaint narrative:
{complaint_text}

Retrieved RG 271 context:
{rag_context}

Generate the internal complaint handling note.
"""
            }
        ]
    )

    return response.choices[0].message.content

def hash_input_text(text: str) -> str:
    """
    Create a one-way hash of the complaint text.

    This allows auditability without storing the raw complaint text
    in the audit log.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_audit_log(
    customer_id: Optional[str],
    complaint_text: str,
    predicted_label: str,
    classification_confidence: float,
    retrieved_rg_refs: list[str],
    human_review_required: bool,
    model_name: str
) -> None:
    """
    Append one API analysis event to the audit log.
    """
    file_exists = AUDIT_LOG_PATH.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "customer_id": customer_id or "",
        "input_hash": hash_input_text(complaint_text),
        "predicted_label": predicted_label,
        "classification_confidence": classification_confidence,
        "retrieved_rg_refs": " | ".join(retrieved_rg_refs),
        "human_review_required": human_review_required,
        "model_name": model_name,
    }

    with open(AUDIT_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
# ------------------------------------------------------------
# Health check endpoint
# ------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "complaint-intelligence-platform",
        "version": "0.2.0",
        "chroma_collection_count": collection.count(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ------------------------------------------------------------
# Main analysis endpoint
# ------------------------------------------------------------

@app.post("/analyze-complaint", response_model=ComplaintAnalysisResponse)
def analyze_complaint(request: ComplaintRequest):
    complaint_text = request.complaint_text

    retrieved_rows = retrieve_rg271_context(
        complaint_text=complaint_text,
        n_results_per_query=5,
        top_k_final=5
    )

    handling_note = generate_handling_note(
        complaint_text=complaint_text,
        retrieved_rows=retrieved_rows
    )

    retrieved_context_response = [
        RetrievedContextItem(
            final_rank=row["final_rank"],
            distance=row["distance"],
            query_name=row["query_name"],
            title=row["title"],
            rg_refs=row["rg_refs"],
            chunk_type=row["chunk_type"],
            text_preview=row["text"][:500]
        )
        for row in retrieved_rows
    ]

    retrieved_rg_refs = [row["rg_refs"] for row in retrieved_rows]
    predicted_label = "PENDING_CLASSIFICATION_MODEL"
    classification_confidence = 0.0
    human_review_required = True
    model_name = "gpt-4.1-mini"

    write_audit_log(
        customer_id=request.customer_id,
        complaint_text=complaint_text,
        predicted_label=predicted_label,
        classification_confidence=classification_confidence,
        retrieved_rg_refs=retrieved_rg_refs,
        human_review_required=human_review_required,
        model_name=model_name
    )

    return ComplaintAnalysisResponse(
        received_at=datetime.now(timezone.utc).isoformat(),
        input_preview=complaint_text[:300],
        predicted_label=predicted_label,
        classification_confidence=classification_confidence,
        retrieved_rg_refs=retrieved_rg_refs,
        retrieved_context=retrieved_context_response,
        human_review_required=human_review_required,
        handling_note=handling_note
    )
