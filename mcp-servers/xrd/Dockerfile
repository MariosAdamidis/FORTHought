FROM python:3.12-slim

WORKDIR /app

# System deps for pymatgen (needs C compiler for some extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV PYTHONUNBUFFERED=1

EXPOSE 9007

CMD ["python", "server.py"]
