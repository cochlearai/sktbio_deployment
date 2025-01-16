#!/bin/bash

read -p "Enter the shm-size (e.g., 10G): " SHM_SIZE

if [ -z "$SHM_SIZE" ]; then
  SHM_SIZE="10G"
  echo "Using default shm-size: $SHM_SIZE"
fi

docker run -d --name multi-model-triton --gpus all --shm-size="$SHM_SIZE" \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $PWD/model:/mnt/model_repository \
  -v $PWD/nemo-model:/mnt/raw_model \
  triton tritonserver \
  --model-repository=/mnt/model_repository --log-verbose=1
