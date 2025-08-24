# Multi-purpose Dockerfile: build for FastAPI service
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run FastAPI. To run Streamlit, override CMD at runtime.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
