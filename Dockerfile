FROM python:3.11-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /mlflow \
    && chown app:app /mlflow
COPY --chown=app:app . .
ENV PYTHONPATH=/app
USER app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "web.app:app"]
