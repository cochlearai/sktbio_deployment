#!/bin/bash

CONTAINER_NAME="multi-model-triton"

docker ps --filter "name=$CONTAINER_NAME" --quiet >/dev/null

if [ $? -eq 0 ]; then
  echo "Stopping container '$CONTAINER_NAME'..."
  docker stop "$CONTAINER_NAME"
else
  echo "Container '$CONTAINER_NAME' is already stopped."
fi

echo "Removing container '$CONTAINER_NAME'..."
docker rm "$CONTAINER_NAME"
