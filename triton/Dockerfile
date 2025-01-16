FROM nvcr.io/nvidia/tritonserver:24.10-py3

RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        transformers \
        torch \
        accelerate \
        torchaudio \
        einops \
        Cython \
        packaging \
        "nemo_toolkit[asr]"
