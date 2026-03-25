# 🩺 Medical RAG + SQL + PDF System

A production-ready Medical AI system that combines natural language interfaces with structured medical data analytics.

## Features

- **Natural Language SQL Queries** - Ask questions in plain English, get SQL results
  - 86+ query types supported across 11 categories
  - Patient registration, visit volume, physician analysis, diagnosis trends, recurrence analysis, payer analysis, staffing, retention, growth, anomaly detection, and data quality
  
- **RAG via FAISS** - Patient-specific medical document retrieval
  - PDF ingestion and embedding
  - Cross-encoder reranking
  - Context-aware chat

- **PostgreSQL Data Warehouse** - Structured medical analytics
  - Star schema with dimension and fact tables
  - ICD-10 diagnosis codes
  - Patient demographics and residence
  - Physician and specialty tracking
  - Payer analysis

- **Ollama LLM** (llama3.2:3b) - On-premise text generation and embeddings

- **PDF Report Generator** - Automated medical reports
  - Patient information
  - Clinical assessment (RAG-generated)
  - Growth & development notes
  - Visit history
  - Follow-up recommendations

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 16GB+ RAM recommended

### Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker ps
```

Services will be available at:
- **Backend API**: http://localhost:8000
- **Streamlit UI**: http://localhost:8501
- **PostgreSQL**: localhost:5433 (internal: 5432)
- **Ollama**: http://localhost:11434

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"healthy"}
```

---

## Usage Examples

### SQL Query Examples

```bash
# Patient registration
curl -X POST http://localhost:8000/api/sql_query/ \
  -H "Content-Type: application/json" \
  -d '{"nl_query": "How many patients were registered between June and July 2025?"}'

# Visit volume comparison
curl -X POST http://localhost:8000/api/sql_query/ \
  -H "Content-Type: application/json" \
  -d '{"nl_query": "How does visit volume compare between 2024 and 2025?"}'

# Top diagnoses
curl -X POST http://localhost:8000/api/sql_query/ \
  -H "Content-Type: application/json" \
  -d '{"nl_query": "What were the top diagnoses between June and July 2025?"}'

# List patients with a condition
curl -X POST http://localhost:8000/api/sql_query/ \
  -H "Content-Type: application/json" \
  -d '{"nl_query": "list patients diagnosed with malaria in 2024"}'

# Recurrence analysis
curl -X POST http://localhost:8000/api/sql_query/ \
  -H "Content-Type: application/json" \
  -d '{"nl_query": "Which diagnoses had the highest recurrence rates between 2024 and 2025?"}'

# Data quality check
curl -X POST http://localhost:8000/api/sql_query/ \
  -H "Content-Type: application/json" \
  -d '{"nl_query": "Were there duplicate patient records detected between January and December 2025?"}'
```

### Query Categories

| Category | Examples |
|----------|----------|
| **Patient Registration** | Registration counts, gender/age distribution, growth rates |
| **Visit Volume** | Total visits, busiest days/months, peak hours |
| **Physician** | Workload distribution, specialty analysis, repeat visits |
| **Diagnosis** | Top diagnoses, disease prevalence, seasonal patterns |
| **Recurrence** | Recurrence rates, high-risk patients, chronic conditions |
| **Payer** | Insurance distribution, self-pay vs insured |
| **Staffing** | Daily load, peak staffing needs, projections |
| **Retention** | 30/60/90-day return rates, follow-up patterns |
| **Growth** | Year-over-year changes, emerging trends |
| **Anomaly** | Sudden spikes, unusual increases |
| **Data Quality** | Missing data, duplicates, inconsistencies |

---

## Database Schema

### Star Schema

```
dim_patient ─────┐
dim_physician ───┼──► fact_patient_visits ◄── dim_date
dim_diagnosis ───┤              │
dim_payer ───────┘              │
                                 ▼
                  fact_recurrence_analysis
```

### Tables

- **dim_patient**: Patient demographics (ID, name, gender, DOB, residence)
- **dim_physician**: Physician info (name, specialty)
- **dim_diagnosis**: ICD-10 codes and descriptions
- **dim_payer**: Insurance/payer information
- **dim_date**: Calendar dimension (year, month, quarter, weekend)
- **fact_patient_visits**: Visit records with foreign keys
- **fact_recurrence_analysis**: Recurrence tracking per patient/diagnosis

---

## Project Structure

```
rag_ollama/
├── app/
│   ├── api/              # FastAPI endpoints
│   ├── agent/            # SQL agent, patient detection, reports
│   ├── core/             # Config, database, errors
│   ├── llm/              # Ollama client, prompts
│   ├── rag/              # Vectorstore, PDF processing
│   └── ui/               # Streamlit app
├── data/                 # PDFs, vectorstore, reports
├── docker/               # Dockerfiles
├── init/                 # Database schema
├── docker-compose.yml
└── .env
```

---

## Configuration

Edit `.env` file:

```env
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=data_warehouse
DB_HOST=data_warehouse
DB_PORT=5432

OLLAMA_MODEL=llama3.2:3b
OLLAMA_URL=http://localhost:11434
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/sql_query/` | POST | Natural language SQL queries |
| `/api/detect_patient/` | POST | Patient identification |
| `/api/chat/` | POST | RAG document chat |
| `/api/upload_pdf/` | POST | Upload PDF documents |
| `/api/report/{patient_id}` | POST | Generate PDF report |

---

## Development

### Rebuild Backend

```bash
docker compose build backend
docker compose up -d backend
```

### View Logs

```bash
docker compose logs -f backend
```

### Stop Services

```bash
docker compose down
```

### Clear Data

```bash
# Remove volumes (WARNING: deletes all data)
docker compose down -v
```

---

## Testing

The system has been tested with **86 different query patterns** across 11 categories:

- Patient Registration: 10 queries
- Visit Volume: 10 queries
- Physician Analysis: 8 queries
- Diagnosis Analysis: 10 queries
- Recurrence Analysis: 10 queries
- Payer Analysis: 7 queries
- Staffing & Operations: 7 queries
- Retention Analysis: 6 queries
- Growth & Trends: 5 queries
- Anomaly Detection: 5 queries
- Data Quality: 8 queries

**All 86 queries pass (100% success rate)**

---

## License

MIT
