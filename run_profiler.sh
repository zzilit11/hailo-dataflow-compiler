#!/bin/bash

# Usage: ./run_model.sh resnet50_v1
MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Please provide a model name. Example: ./run_model.sh resnet50_v1"
    exit 1
fi

# Define paths
HEF_PATH="./hef_models/${MODEL_NAME}.hef"
HAR_PATH="./har_models/${MODEL_NAME}_compiled.har"

# Output filenames
RUNTIME_JSON="runtime_data_${MODEL_NAME}.json"
HTML_REPORT="${MODEL_NAME}_compiled_model.html"

# Target directory
PROFILING_DIR="profiling"

# Create profiling directory if it doesn't exist
if [ ! -d "$PROFILING_DIR" ]; then
    mkdir -p "$PROFILING_DIR"
fi

# Execute commands
hailortcli run2 -m raw_sync measure-fw-actions set-net "${HEF_PATH}"
hailo profiler "${HAR_PATH}" --runtime-data "${RUNTIME_JSON}"

# Move generated files to profiling directory
echo "Moving output files to ${PROFILING_DIR}..."

if [ -f "$RUNTIME_JSON" ]; then
    mv "$RUNTIME_JSON" "${PROFILING_DIR}/"
fi

if [ -f "$HTML_REPORT" ]; then
    mv "$HTML_REPORT" "${PROFILING_DIR}/"
fi

echo "Done."