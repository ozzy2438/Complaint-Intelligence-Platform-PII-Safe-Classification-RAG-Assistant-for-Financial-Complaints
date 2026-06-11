

# Complaint Intelligence Platform

**PII-safe classification and retrieval-augmented complaint handling assistant for financial services complaints.**

This project demonstrates an end-to-end complaint intelligence workflow for regulated financial services environments. It combines complaint text processing, PII redaction, product classification, ASIC RG 271 regulatory retrieval, RAG-based internal handling notes, human-review controls, and a FastAPI service layer.

The goal is not to automate final complaint decisions. The goal is to support internal triage teams with faster, more consistent, and auditable complaint analysis.

---

## 1. Project Summary

Financial institutions receive large volumes of customer complaints across products such as debt collection, credit cards, bank accounts, loans, and money transfers. These complaints often require careful triage, regulatory awareness, timely handling, and human review.

This project builds a proof-of-concept platform that:

- redacts personal information before downstream processing,
- classifies complaint narratives into product categories,
- retrieves relevant ASIC RG 271 internal dispute resolution guidance,
- generates internal complaint handling notes grounded in retrieved regulatory context,
- flags cases for human review,
- exposes the workflow through a FastAPI endpoint,
- records audit logs without storing raw complaint text.

---

## 2. Architecture

```text
Complaint narrative
        ↓
PII redaction layer
        ↓
Baseline product classification
        ↓
Multi-query RG 271 retrieval
        ↓
Chroma vector database
        ↓
Retrieved regulatory context
        ↓
LLM-generated internal handling note
        ↓
Human review flag + audit log
        ↓
FastAPI JSON response
```

---

## 3. Data Sources

### Complaint data

- CFPB Consumer Complaint Database
- Used for complaint narratives and labelled product categories
- Source complaint narratives are already partially anonymised by CFPB
- A separate PII redaction layer is still included to demonstrate production-grade privacy controls

### Regulatory knowledge base

- ASIC Regulatory Guide 271: Internal Dispute Resolution
- Used as the RAG knowledge base for internal dispute resolution context
- The document was chunked using an agentic RG-aware chunking strategy

---

## 4. Key Components

### PII redaction

Microsoft Presidio is used to detect and mask personal information in complaint narratives before modelling or retrieval.

This demonstrates a responsible AI control that would be required for raw internal banking data.

### Baseline classification

A TF-IDF + Logistic Regression baseline model was trained to classify complaint narratives into product categories.

The baseline is intentionally simple and interpretable. It provides a reference point before introducing more complex NLP or transformer-based models.

Example baseline result:

```text
Accuracy: approximately 0.81
Weighted F1: approximately 0.81
```

### Error and confidence analysis

The baseline model was evaluated with:

- classification report,
- confusion matrix,
- error pair analysis,
- confidence threshold review,
- human-review routing logic.

This helped identify common confusion patterns such as credit card vs checking account and loan-related category overlap.

### Agentic RG 271 chunking

Several chunking strategies were tested:

- recursive chunking,
- semantic chunking,
- paragraph-aware chunking,
- RG-aware chunking,
- agentic RG-aware chunking.

The agentic approach produced the strongest results because it preserved:

- complete words,
- complete sentences,
- RG references,
- meaningful titles,
- chunk types,
- standalone regulatory context.

The final RG 271 chunks were saved with QA flags for downstream retrieval testing.

### Chroma vector retrieval

The cleaned RG 271 chunks were embedded with `all-MiniLM-L6-v2` and stored in a local Chroma vector database.

Multi-query retrieval was used instead of a single raw complaint query. This improved retrieval quality by searching from several regulatory angles:

- complaint definition,
- IDR process and response,
- IDR timeframes,
- dispute / debt collection context,
- AFCA escalation.

### RAG complaint handling note

Retrieved RG 271 context is passed to an LLM to generate an internal complaint handling note.

The output includes:

1. complaint context,
2. relevant RG 271 guidance,
3. suggested handling considerations,
4. human review flag.

The prompt explicitly prevents legal advice and requires the model to use only retrieved context.

---

## 5. API Layer

The project exposes a FastAPI service with two endpoints:

```text
GET /health
POST /analyze-complaint
```

Example request:

```json
{
  "complaint_text": "I am disputing a debt collection account. The collector has not provided documents proving that I owe the debt, and they continue to contact me despite my dispute.",
  "customer_id": "demo_customer_001"
}
```

Example response fields:

```json
{
  "received_at": "2026-06-11T07:36:57Z",
  "input_preview": "I am disputing a debt collection account...",
  "predicted_label": "PENDING_CLASSIFICATION_MODEL",
  "classification_confidence": 0.0,
  "retrieved_rg_refs": ["RG 271.90", "RG 271.49", "RG 271.50"],
  "retrieved_context": [],
  "human_review_required": true,
  "handling_note": "Internal Complaint Handling Note..."
}
```

---

## 6. Audit Logging

The API includes an audit logging design so each model-assisted analysis can be traced without storing raw complaint text.

The audit log records:

- timestamp,
- customer or case ID,
- one-way input hash,
- predicted label,
- classification confidence,
- retrieved RG references,
- human review flag,
- model name.

This supports responsible AI, traceability, and compliance-style review.

---

## 7. Responsible AI Controls

This project intentionally includes controls that are important in regulated financial services settings:

- PII redaction before downstream processing,
- no raw complaint text stored in audit logs,
- retrieved regulatory context retained for review,
- human review required for final decision-making,
- prompt rules preventing legal advice,
- generated notes treated as internal triage aids only,
- clear limitations documented.

---

## 8. Current Limitations

This is a demonstration system, not a production banking application.

Current limitations include:

- CFPB data is US-based while RG 271 is Australian regulation,
- classification is currently a baseline model,
- RAG retrieval quality depends on chunk quality and query design,
- generated notes require human validation,
- no authentication layer is included,
- no production monitoring dashboard is included,
- no final legal or customer-facing advice is produced.

---

## 9. Future Improvements

Planned improvements:

- save and load the trained classification model with `joblib`,
- connect classification output to the FastAPI response,
- add DistilBERT or transformer-based classification,
- add a lightweight Streamlit or React demo interface,
- improve retrieval with hybrid keyword + vector search,
- add RAGAS evaluation for faithfulness and answer relevance,
- add Docker deployment,
- deploy to Azure Container Apps or AWS App Runner,
- add authentication and request-level monitoring.

---

## 10. Tech Stack

- Python
- pandas
- scikit-learn
- Microsoft Presidio
- sentence-transformers
- ChromaDB
- OpenAI API
- FastAPI
- Uvicorn
- python-dotenv
- Jupyter notebooks

---

## 11. How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

Run the API:

```bash
uvicorn src.api.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 12. Demo Workflow

1. Open the FastAPI Swagger UI.
2. Run `GET /health` to confirm the API is live.
3. Run `POST /analyze-complaint` with a sample complaint narrative.
4. Review the returned RG 271 references, retrieved context, handling note, and human review flag.
5. Check the audit log output in the `reports/` folder.

---

## 13. Example Use Case

A complaints team receives a disputed debt collection complaint. The platform redacts sensitive information, retrieves relevant RG 271 guidance about IDR timeframes and AFCA escalation, and generates an internal note recommending that the team review debt validation documents, respond within the relevant complaint handling timeframe, and keep the case under human review.

---

## 14. Positioning for Job Applications

This project demonstrates practical skills across:

- NLP and text classification,
- responsible AI and privacy controls,
- retrieval-augmented generation,
- regulatory document processing,
- vector databases,
- API development,
- auditability and human-in-the-loop design,
- financial services domain awareness.

It is designed as a realistic portfolio project for data, AI, analytics, and technical product roles in regulated environments.

---

## 15. Disclaimer

This project is for educational and portfolio demonstration purposes only. It does not provide legal advice, financial advice, or final complaint determinations. All generated outputs require human review before use in any real complaint handling process.