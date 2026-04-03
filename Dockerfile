FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime/system deps:
# - poppler-utils: pdf2image backend
# - pandoc: markdown -> pdf conversion
# - libgl1/libglib2.0-0: OpenCV headless runtime requirements
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      poppler-utils \
      pandoc \
      libgl1 \
      libglib2.0-0 \
      libmagic1 \
      curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
