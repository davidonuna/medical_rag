FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (Streamlit needs app/ui, utils, etc.)
COPY ./app /app/app

EXPOSE 8501

CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
