#!/bin/bash
set -e

IMAGE_NAME="spark4-unsloth"

echo "Building Docker image..."
docker build -t $IMAGE_NAME .

echo ""
echo "Starting container with GPU support..."
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    --shm-size=16g \
    $IMAGE_NAME
