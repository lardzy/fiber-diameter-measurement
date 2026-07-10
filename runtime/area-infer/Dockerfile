FROM python:3.11-slim

WORKDIR /app

ARG TORCH_VARIANT=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY requirements.torch.cpu.txt ./
COPY requirements.torch.gpu.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$TORCH_VARIANT" = "gpu" ]; then \
         pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 -r requirements.torch.gpu.txt; \
       else \
         pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements.torch.cpu.txt; \
       fi

RUN groupadd --gid 10001 area-infer \
    && useradd --uid 10001 --gid area-infer --create-home --home-dir /home/area-infer --shell /usr/sbin/nologin area-infer \
    && mkdir -p /opt/area_weights \
    && chown area-infer:area-infer /opt/area_weights

COPY --chown=area-infer:area-infer app ./app
COPY --chown=area-infer:area-infer vendor ./vendor

ENV PYTHONPATH=/app:/app/vendor/yolact \
    HOME=/home/area-infer \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
ENV AREA_ALLOW_ANONYMOUS_DEV=0 \
    AREA_REQUIRE_TRUSTED_WEIGHTS=1 \
    AREA_VERIFY_TRUSTED_WEIGHTS=1 \
    AREA_MAX_CACHED_MODELS=2 \
    AREA_MAX_REQUEST_BYTES=67108864 \
    AREA_MAX_IMAGE_BYTES=50331648 \
    AREA_MAX_IMAGE_PIXELS=50000000 \
    AREA_MAX_MASK_WORKING_BYTES=1610612736 \
    AREA_MAX_CONCURRENT_INFER=1

EXPOSE 9001

USER 10001:10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9001/ready', timeout=3).read()"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9001", "--limit-concurrency", "4"]
