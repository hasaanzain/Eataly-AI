FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/

# Run as a non-root user; the knowledge base is downloaded from S3 into /app at startup.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Port 8501 matches the load balancer target group and task definition in infra/aws/app.yaml.
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/api/health')"

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8501", "--timeout-graceful-shutdown", "20"]
