# Dockerfile for RunPod Serverless GPU (flash-boot ready)
FROM runpod/serverless:gpu-cuda12.1.1

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# ---- Cache BiRefNet model at build time ----
ARG HF_MODEL_ID=ZhengPeng7/BiRefNet_lite
ENV MODEL_LOCAL_DIR=/models/birefnet
RUN mkdir -p $MODEL_LOCAL_DIR

RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os
repo = os.environ.get("HF_MODEL_ID", "ZhengPeng7/BiRefNet_lite")
dst = os.environ.get("MODEL_LOCAL_DIR", "/models/birefnet")
snapshot_download(repo_id=repo, local_dir=dst, local_dir_use_symlinks=False)
print("✅ Model cached at", dst)
PY

# Stay fully offline at runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy app
COPY handler.py .

# RunPod entrypoint
CMD ["python", "-m", "runpod.serverless.workers.python", "--handler-path", "handler.py", "--handler-name", "rp_handler"]
