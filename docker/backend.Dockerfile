FROM python:3.11-slim

WORKDIR /app

# FIX #2 — Install PDF + libmagic + poppler-utils
RUN apt-get update && \
    apt-get install -y \
        gcc \
        libpq-dev \
        build-essential \
        libffi-dev \
        libfreetype6-dev \
        poppler-utils \
        libmagic1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY ./app /app/app
COPY ./run_agent.py /app/run_agent.py

# Ensure data directories exist
RUN mkdir -p /app/data/pdfs /app/data/reports /app/data/vectorstore

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
