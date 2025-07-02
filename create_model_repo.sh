#!/usr/bin/env bash

# Exit on error
set -e

# Define your paths
MODEL_REPO_DIR="model_repository"
MODEL_NAME="faster_rcnn"
MODEL_VERSION_DIR="${MODEL_REPO_DIR}/${MODEL_NAME}/1"

# Path to your exported ONNX model
# You must set this to your actual model path
SRC_ONNX_PATH="/workspace/output_models/model.onnx"

# Create directories
mkdir -p "${MODEL_VERSION_DIR}"

# Copy the model
cp "${SRC_ONNX_PATH}" "${MODEL_VERSION_DIR}/model.onnx"

# Copy the model
cp ./config.pbtxt "${MODEL_REPO_DIR}/${MODEL_NAME}/config.pbtxt"

echo "✅ Model repository created successfully in: ${MODEL_REPO_DIR}"


