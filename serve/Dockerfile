# chunklaya as a System One endpoint. Built from this checkout, with the Laya
# weights baked in, so a container start never downloads anything.
#
# Build from the repository root:  docker build -f serve/Dockerfile .
#
# The official PyTorch image ships Python 3.12 (this package needs >= 3.12)
# with CUDA 12.6, which any host on driver 12.6 or newer accepts. laya is
# pinned to the version proven against this checkpoint revision; the package
# itself is installed --no-deps so its `torch>=2.14` floor cannot pull a second
# torch over the image's.
FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime

ARG LAYA_REVISION=1c5edc17a7acd8701df6fc341c0d179f1c62c982
ARG GIT_SHA=unknown

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 USE_TF=0 TOKENIZERS_PARALLELISM=false

RUN pip install "fastapi==0.116.1" "uvicorn[standard]==0.35.0" "laya==0.3.4" "transformers==4.56.2" "huggingface_hub>=0.20"

COPY pyproject.toml README.md /src/
COPY chunklaya/ /src/chunklaya/
RUN pip install --no-deps /src \
 && python -c "import chunklaya, laya; print('chunklaya', chunklaya.__version__, 'laya', laya.__version__)"

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('convaiinnovations/laya', revision='${LAYA_REVISION}')"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    LAYA_REVISION=${LAYA_REVISION} LAYA_SUBFOLDER=multilingual LAYA_DEVICE=cuda \
    CHUNKLAYA_GIT_SHA=${GIT_SHA}
LABEL org.opencontainers.image.source="https://github.com/myxamediyar/chunklaya" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app
COPY serve/service.py /app/service.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One process: the model lives in it. Concurrency is the service's own admission, not worker count.
CMD ["uvicorn", "service:api", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info", "--no-access-log"]
