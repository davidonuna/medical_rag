# Medical RAG System - Documentation

## Overview

This is a production-ready Medical AI system that combines:
- **RAG (Retrieval-Augmented Generation)** via FAISS vectorstore per patient
- **PostgreSQL data warehouse** for structured medical data
- **Ollama LLM** (llama3.2:3b) for text generation and embeddings
- **FastAPI backend** for REST API endpoints
- **Streamlit frontend** for user interface
- **SQL Agent** for natural language to SQL query translation
- **Automated PDF Report Generator** for patient medical reports

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI (Port 8501)                   │
│  • Patient Detection  • RAG Chat  • SQL Query  • PDF Upload     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP API
┌──────────────────────────────▼──────────────────────────────────┐
│                    FastAPI Backend (Port 8000)                  │
│  • /api/chat           - RAG query endpoint                    │
│  • /api/detect_patient - Patient identification               │
│  • /api/upload_pdf     - PDF ingestion                         │
│  • /api/sql_query     - Natural language SQL (86+ query types)│
│  • /api/report        - PDF report generation                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
         ┌────────────────────┼──────────────────────┐
         ▼                    ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   PostgreSQL  │    │   FAISS Vector  │    │   Ollama LLM     │
│   Database    │    │     Store       │    │   (Generation)   │
│   (Analytics) │    │  (RAG Context)  │    │   (Embeddings)   │
└───────────────┘    └─────────────────┘    └──────────────────┘
```

---

## Directory Structure

```
rag_ollama/
├── app/
│   ├── api/                  # FastAPI endpoints
│   │   ├── main.py          # Main application with lifespan
│   │   ├── router.py        # SQL/Report routes (legacy endpoint)
│   │   ├── models.py        # Pydantic models
│   │   └── report.py        # Report endpoints
│   │
│   ├── agent/               # AI agents
│   │   ├── patient_detection.py    # Patient identification
│   │   ├── sql_interpreter.py      # NL→SQL translation (legacy)
│   │   ├── sql_agent.py           # Main SQL agent (LangGraph-based)
│   │   ├── sql_tool.py             # SQL execution
│   │   ├── report_generator.py     # PDF report generation
│   │   └── report_agent.py         # LangGraph report agent
│   │
│   ├── core/               # Core utilities
│   │   ├── config.py       # Settings management
│   │   ├── database.py     # PostgreSQL connection pool
│   │   ├── errors.py      # Custom exceptions
│   │   └── logger.py      # Logging configuration
│   │
│   ├── llm/               # LLM integration
│   │   ├── ollama_client.py     # Ollama API client
│   │   └── prompt_templates.py  # All prompts
│   │
│   ├── rag/               # RAG components
│   │   ├── rag_pipeline.py          # Main RAG logic
│   │   ├── pdf_ingestor.py         # PDF processing
│   │   ├── vectorstore_manager.py  # FAISS management
│   │   ├── reranker.py             # Cross-encoder reranker
│   │   └── singletons.py           # Global singletons
│   │
│   ├── ui/                # Streamlit UI
│   │   └── streamlit_app.py
│   │
│   └── utils/             # Utilities
│       ├── pdf_utils.py
│       └── timing.py
│
├── data/                  # Data storage
│   ├── vectorstore/       # FAISS index + metadata
│   ├── reports/           # Generated PDF reports
│   ├── metadata/          # Extracted PDF metadata
│   └── pdfs/             # Uploaded PDFs
│
├── docker/               # Dockerfiles
│   ├── backend.Dockerfile
│   ├── streamlit.Dockerfile
│   └── ollama.Dockerfile
│
├── init/                 # Database initialization
│   └── 01-create-schema.sql
│
├── docker-compose.yml     # Docker Compose configuration
├── .env                  # Environment variables
└── requirements.txt       # Python dependencies
```

---

## Database Schema

### Dimension Tables

```sql
-- Patient dimension
dim_patient(
    patient_id TEXT PRIMARY KEY,      -- Format: NCH-XXXXX
    first_name VARCHAR,
    last_name VARCHAR,
    gender CHAR(1),                   -- 'M' or 'F'
    dob DATE,
    age INT,
    contact VARCHAR,
    residence VARCHAR,                 -- Patient's residential area
    registered_by VARCHAR,
    created_at TIMESTAMP,
    full_name TEXT GENERATED          -- lowercase first_name + last_name
)

-- Physician dimension
dim_physician(
    physician_id INT PRIMARY KEY,
    name VARCHAR,
    specialty VARCHAR
)

-- Diagnosis dimension
dim_diagnosis(
    diagnosis_id SERIAL PRIMARY KEY,
    icd10_code VARCHAR UNIQUE,
    description VARCHAR               -- Diagnosis description (e.g., "Plasmodium falciparum malaria")
)

-- Payer dimension
dim_payer(
    payer_id SERIAL PRIMARY KEY,
    payer_name VARCHAR UNIQUE,
    payer_type VARCHAR               -- e.g., "Insurance", "Self-pay"
)

-- Date dimension
dim_date(
    date_id SERIAL PRIMARY KEY,
    calendar_date DATE UNIQUE,
    year INT,
    month INT,
    day INT,
    day_of_week INT,                -- 1=Monday, 7=Sunday
    week_of_month INT,
    is_weekend BOOLEAN,
    year_month VARCHAR(7),           -- Format: YYYY-MM
    quarter INT                      -- 1-4
)
```

### Fact Tables

```sql
-- Patient visits fact
fact_patient_visits(
    visit_id SERIAL PRIMARY KEY,
    patient_id VARCHAR REFERENCES dim_patient,
    physician_id INT REFERENCES dim_physician,
    diagnosis_id INT REFERENCES dim_diagnosis,
    payer_id INT REFERENCES dim_payer,
    date_id INT REFERENCES dim_date,
    visit_timestamp TIMESTAMP,
    visit_hour INT,                  -- Hour of visit (0-23)
    created_at TIMESTAMP
)

-- Recurrence analysis fact
fact_recurrence_analysis(
    patient_id VARCHAR REFERENCES dim_patient,
    diagnosis_id INT REFERENCES dim_diagnosis,
    recurrence_count INT,
    first_occurrence_date DATE,
    last_occurrence_date DATE,
    created_at TIMESTAMP,
    PRIMARY KEY (patient_id, diagnosis_id)
)
```

---

## SQL Agent

### Architecture

The SQL Agent has two components:

1. **`sql_agent.py`** - Main agent using pattern-based interpretation with fallback to LLM
2. **`sql_interpreter.py`** - Legacy interpreter with 86+ query type patterns

### Key Features

- **SQL Validator**: Blocks dangerous operations (DROP, DELETE, UPDATE, INSERT, etc.)
- **Query Cache**: 60-second TTL caching
- **Error Handler**: User-friendly error messages
- **Intent Detection**: 86+ query types supported
- **Diagnosis Extraction**: Supports multi-word diagnoses, hyphens, and periods

### Supported Query Types

#### Patient Registration
| Query Type | Example |
|------------|---------|
| Count | "How many patients were registered between June and July 2025?" |
| Period comparison | "How does patient registration compare between 2024 and 2025?" |
| Gender distribution | "What was the gender distribution of newly registered patients?" |
| Age distribution | "What was the age distribution of patients registered between Q1 and Q2?" |
| By residence | "Which residential areas registered the most patients?" |
| Growth rate | "What was the monthly growth rate in patient registrations?" |
| By staff | "Which staff members registered the most patients?" |
| Pediatric proportion | "What proportion of registered patients were pediatric?" |
| Gender ratio change | "Did the gender ratio of new registrations change?" |

#### Visit Volume
| Query Type | Example |
|------------|---------|
| Total visits | "How many total visits occurred between June and July 2025?" |
| Period comparison | "How does visit volume compare between 2024 and 2025?" |
| Average per patient | "What was the average number of visits per patient?" |
| Multiple visits | "What percentage of patients had more than one visit?" |
| Busiest days | "Which days of the week were busiest?" |
| Busiest month | "Which month had the highest visit volume?" |
| Quarterly change | "How did quarterly visit volume change?" |
| Weekend percentage | "What percentage of visits occurred on weekends?" |
| Peak hours | "What were the peak consultation hours?" |
| YoY growth | "What was the year-over-year visit growth rate?" |

#### Physician Analysis
| Query Type | Example |
|------------|---------|
| Visits per physician | "How many visits were handled by each physician?" |
| Workload change | "How did physician workload change between 2024 and 2025?" |
| Top physician | "Which physician saw the highest number of patients?" |
| Specialty distribution | "How did visit distribution by specialty change?" |
| Specialty growth | "Which specialty experienced the greatest growth?" |
| Repeat visits | "What proportion of repeat visits did each physician manage?" |
| Load distribution | "How evenly was patient load distributed among physicians?" |
| Peak hours | "What were the peak working hours per physician?" |

#### Diagnosis Analysis
| Query Type | Example |
|------------|---------|
| Top diagnoses | "What were the top diagnoses between June and July 2025?" |
| ICD-10 frequency | "How did ICD-10 code frequency change?" |
| By gender | "How did disease prevalence vary by gender?" |
| By age group | "How did disease prevalence vary by age group?" |
| By residence | "Which diagnoses were most common by residence?" |
| Monthly trend | "How did specific diagnoses trend monthly?" |
| Seasonal patterns | "Were there seasonal disease patterns observed?" |
| Significant increase | "Which diagnoses increased significantly?" |
| Pediatric | "What were the most common pediatric diagnoses?" |
| Chronic burden | "How did chronic disease burden change?" |

#### Recurrence Analysis
| Query Type | Example |
|------------|---------|
| Highest recurrence | "Which diagnoses had the highest recurrence rates?" |
| Recurring patients | "How many patients experienced recurring diagnoses?" |
| Average count | "What was the average recurrence count per diagnosis?" |
| Time span | "What was the average time between first and last occurrence?" |
| Pattern change | "How did recurrence patterns change between Q1 and Q2?" |
| Chronic contributions | "Which chronic conditions contributed most to visits?" |
| By age group | "How did recurrence rates vary by age group?" |
| By physician | "How did recurrence rates vary by physician?" |
| By payer | "How did recurrence vary by payer type?" |
| High-risk | "How many high-risk patients were identified?" |

#### Payer Analysis
| Query Type | Example |
|------------|---------|
| Distribution | "What was the distribution of visits by payer type?" |
| Top payer | "Which payer accounted for the highest visits?" |
| Self-pay vs insured | "How did self-pay versus insured visit ratios change?" |
| Monthly trend | "How did payer utilization trend monthly?" |
| Common diagnoses | "Which diagnoses were most common under each payer?" |
| Recurrence | "Did recurrence rates differ by payer?" |
| Private vs public | "How did private versus public insurance visit volumes compare?" |

#### Staffing & Operations
| Query Type | Example |
|------------|---------|
| Daily load | "What was the average daily patient load?" |
| Peak staffing | "Which months required the highest staffing levels?" |
| Peak days | "What were the peak operational days?" |
| Specialty capacity | "Which specialties operated at highest capacity?" |
| Underutilization | "Were there underutilized physicians?" |
| Staffing adjustment | "How should staffing be adjusted based on hourly patterns?" |
| Projection | "What is the projected patient load for the next quarter?" |

#### Retention Analysis
| Query Type | Example |
|------------|---------|
| 30-day return | "Of patients registered between January and March 2025, how many returned within 30 days?" |
| 3-month rate | "What was the three-month retention rate for patients registered between June and July?" |
| Follow-up time | "How long after initial diagnosis did patients return for follow-up?" |
| 60-day proportion | "What proportion of first-time patients returned within 60 days?" |
| By diagnosis | "How did retention vary by diagnosis?" |
| By payer | "How did retention vary by payer type?" |

#### Growth & Trends
| Query Type | Example |
|------------|---------|
| Volume increase | "Did overall patient volume increase between 2024 and 2025?" |
| Geographic growth | "Which residential areas showed the highest growth?" |
| Workload drivers | "Which diagnoses emerged as major workload drivers?" |
| Specialty growth | "Which specialties experienced the fastest growth?" |
| Chronic proportion | "How did the proportion of chronic conditions change?" |

#### Anomaly Detection
| Query Type | Example |
|------------|---------|
| Diagnosis spikes | "Were there sudden spikes in specific diagnoses?" |
| Pediatric respiratory | "Was there an unusual increase in pediatric respiratory cases?" |
| Weekend volumes | "Did weekend visit volumes rise unexpectedly?" |
| Physician fluctuations | "Were there abnormal fluctuations in physician visit counts?" |
| Recurrence spikes | "Did recurrence rates for certain diseases increase sharply?" |

#### Data Quality
| Query Type | Example |
|------------|---------|
| Patients without visits | "Were there patients registered without visits?" |
| Visits without diagnosis | "Were there visits recorded without diagnoses?" |
| Missing physician | "Were there visits without assigned physicians?" |
| Missing payer | "Were there missing payer assignments?" |
| Age/DOB inconsistency | "Were there inconsistencies between age and date of birth?" |
| Duplicates | "Were duplicate patient records detected?" |
| Gender distribution | "Were there abnormal gender distributions?" |
| Timestamp consistency | "Were visit timestamps inconsistent with calendar dates?" |

### List Patient Queries

Special support for listing individual patients:

| Pattern | Example |
|---------|---------|
| "list patients with X" | "list patients with malaria" |
| "show patients with X" | "show patients with hypertension" |
| "find patients with X" | "find patients with diabetes" |
| "display patients with X" | "display patients diagnosed with sickle cell" |
| "get all patients with X" | "get all patients with asthma" |

Supports multi-word diagnoses: "plasmodium falciparum malaria", "sickle-cell anaemia"

---

## Patient Detection

Three-stage patient identification:

1. **Regex**: Direct `NCH-\d+` pattern matching
2. **Trigram Search**: PostgreSQL trigram similarity (threshold ≥ 0.4)
3. **LLM Extraction**: Ollama-based patient ID extraction

```python
result = await detect_patient_async("Patient John Smith")
# Returns: {
#   "patient_id": "NCH-12345",
#   "confidence": 0.85,
#   "source": "trigram",
#   "first_name": "John",
#   "last_name": "Smith"
# }
```

---

## RAG Pipeline

### Flow

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Index
                                                        ↓
User Query → Embed Query → Similarity Search → Rerank → LLM Generate → Response
```

### Key Classes

#### `VectorStoreManager` - Manages FAISS vector index

```python
vectorstore = VectorStoreManager()
await vectorstore.add_patient_documents(patient_id="NCH-12345", documents=["clinical text..."])
results = await vectorstore.retrieve_raw("What medications?", top_n=15)
```

#### `RAGPipeline` - Main RAG orchestration

```python
rag = RAGPipeline()
result = await rag.query_async(question="What medications?", patient_id="NCH-12345")
# Returns: {"answer": "...", "citations": [...]}
```

---

## PDF Report Generator

Generates medical reports combining SQL data + RAG summaries:

```python
generator = MedicalReportGenerator()
pdf_path = await generator.generate("NCH-12345")
```

**Report Sections:**
1. Patient Information (ID, name, DOB, calculated age)
2. Clinical Assessment (RAG-generated summary)
3. Growth & Development (age-appropriate notes)
4. Visit History (from database)
5. Clinical Impression & Plan
6. Follow-Up Recommendations

---

## API Endpoints

### Backend (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/detect_patient/` | POST | Identify patient from text |
| `/api/chat/` | POST | RAG query |
| `/api/upload_pdf/` | POST | Upload & process PDF |
| `/api/report/{patient_id}` | POST | Generate PDF report |
| `/api/report/status/{patient_id}` | GET | Report generation status |
| `/api/report/download/{patient_id}` | GET | Download PDF report |
| `/api/sql_query/` | POST | Natural language SQL (86+ types) |

### Streamlit UI Tabs

1. **Patient Detection** - Identify patients by ID or name
2. **RAG Chat** - Query medical documents
3. **SQL Query** - Natural language database queries
4. **Upload PDF** - Add documents to vectorstore
5. **Generate Report** - Create PDF reports

---

## Running the Application

### With Docker Compose

```bash
# Start all services
docker compose up -d

# Services:
# - PostgreSQL: localhost:5433 (mapped to 5432 internally)
# - Ollama: localhost:11434
# - Backend: localhost:8000
# - Streamlit: localhost:8501

# View logs
docker compose logs -f

# Rebuild after changes
docker-compose build backend && docker-compose up -d backend
```

### Local Development

```bash
# Backend
uvicorn app.api.main:app --reload --port 8000

# Streamlit
streamlit run app/ui/streamlit_app.py

# Ollama (pull model)
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

---

## Environment Variables

Create `.env` file:

```env
# PostgreSQL / Data Warehouse
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=data_warehouse
DB_HOST=data_warehouse
DB_PORT=5432

# Ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_URL=http://localhost:11434

# File paths
PDF_UPLOAD_PATH=./data/pdfs
VECTORSTORE_PATH=./data/vectorstore
REPORT_PATH=./data/reports
```

---

## Dependencies

```
fastapi>=0.100.0
streamlit>=1.30.0
asyncpg>=0.29.0
faiss-cpu>=1.7.0
pdfminer.six>=20221105
reportlab>=4.0.0
aiohttp>=3.9.0
sentence-transformers>=2.2.0
pydantic>=2.0.0
python-multipart>=0.0.6
```

---

## Error Handling

Custom exceptions in `app/core/errors.py`:

- `DatabaseError` - Database failures
- `PDFError` - PDF generation failures
- `VectorStoreError` - Vector operations

---

## Testing

The system has been tested with 86 different query patterns across 11 categories:

- Patient Registration: 10 queries
- Visit Volume: 10 queries
- Physician: 8 queries
- Diagnosis: 10 queries
- Recurrence: 10 queries
- Payer: 7 queries
- Staffing: 7 queries
- Retention: 6 queries
- Growth: 5 queries
- Anomaly: 5 queries
- Data Quality: 8 queries

**All 86 queries pass (100% success rate)**

---

## Changelog

### Version 1.1.0 (2026-03-20)

**New Features:**
- Enhanced diagnosis extraction supporting multi-word diagnoses ("sickle cell disease", "plasmodium falciparum malaria")
- Support for hyphens and periods in diagnosis names
- Default 3-year range for list queries without explicit year
- Fixed LEFT JOIN year filter handling

**Bug Fixes:**
- Fixed "with malaria" capturing issue (was capturing "with malaria" instead of "malaria")
- Fixed LEFT JOIN with date filtering exclusion bug

**Query Improvements:**
- 86 query types now supported across 11 categories
- Better handling of question phrasings (How many, What was, Which, etc.)
- Period format variations (between, for, vs)
