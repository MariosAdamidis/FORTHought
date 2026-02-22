FROM python:3.11-slim

# Install deps
RUN pip install --no-cache-dir fastapi uvicorn && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    update-ca-certificates

# Add Chart.js locally (no CDN at runtime)
RUN mkdir -p /app/static && \
    curl -fsSL -o /app/static/chart.umd.min.js \
    https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js && \
    apt-get purge -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 9001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9001"]
