FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
# default = web gateway; model services override CMD via docker-compose
CMD ["gunicorn", "-b", "0.0.0.0:5000", "web.app:app"]
