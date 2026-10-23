FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

RUN useradd --create-home --uid 10001 margin \
    && mkdir -p /app/.cache/huggingface \
    && chown -R margin:margin /app

COPY --chown=margin:margin \
    api.py config.py generate.py providers.py search.py rerank.py text_units.py ./

COPY --chown=margin:margin static/ ./static/
COPY --chown=margin:margin data/processed/chunks.json ./data/processed/chunks.json
COPY --chown=margin:margin data/processed/embeddings.npy ./data/processed/embeddings.npy

USER margin

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

