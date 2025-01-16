#!/bin/bash

CONTAINER_NAME="multi-model-triton"

docker ps --filter "name=$CONTAINER_NAME" --quiet >/dev/null
if [ $? -ne 0 ]; then
    echo "Container '$CONTAINER_NAME' is not running."
    exit 1
fi

docker logs -f "$CONTAINER_NAME"
