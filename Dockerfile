FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG TORCH_INDEX_URL=
ARG TORCH_VERSION=
ARG TORCHVISION_VERSION=
ARG REQUIREMENTS_FILE=requirements.txt

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

COPY requirements*.txt /app/
RUN pip install --upgrade pip && \
    if [ -n "$TORCH_INDEX_URL" ]; then \
      pip install --index-url "$TORCH_INDEX_URL" "torch${TORCH_VERSION:+==$TORCH_VERSION}" "torchvision${TORCHVISION_VERSION:+==$TORCHVISION_VERSION}"; \
    else \
      pip install "torch${TORCH_VERSION:+==$TORCH_VERSION}" "torchvision${TORCHVISION_VERSION:+==$TORCHVISION_VERSION}"; \
    fi && \
    pip install -r "/app/${REQUIREMENTS_FILE}"

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
