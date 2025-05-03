#!/bin/bash

# Script to build and run a Streamlit Docker container with GPU support
# Usage: ./run_streamlit_gpu.sh [image_name] [container_name] [port]

# Default values
IMAGE_NAME=${1:-"streamlit-gpu-app"}
CONTAINER_NAME=${2:-"streamlit-container"}
PORT=${3:-"8501"}

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo "Stopping any existing container with the same name..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "Running container with GPU support..."
docker run --gpus all \
  --name $CONTAINER_NAME \
  -p $PORT:8501 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -it \
  $IMAGE_NAME

echo "Container started!"
echo "Access your Streamlit app at: http://localhost:$PORT"
echo ""
echo "To view container logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "To stop the container:"
echo "  docker stop $CONTAINER_NAME"