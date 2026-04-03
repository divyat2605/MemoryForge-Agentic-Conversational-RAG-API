FROM python:3.11-slim
 
WORKDIR /app
 
# system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# app code
COPY . .
 
# data dir for persisted BM25 index
RUN mkdir -p /app/data
 
EXPOSE 8000
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
 