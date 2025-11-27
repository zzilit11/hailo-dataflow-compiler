#!/bin/bash

# Usage: ./run_model.sh resnet50_v1
MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Please provide a model name. Example: ./run_model.sh resnet50_v1"
    exit 1
fi

HEF_PATH="./hef_models/${MODEL_NAME}.hef"
HAR_PATH="./har_models/${MODEL_NAME}_compiled.har"
RUNTIME_JSON="runtime_data_${MODEL_NAME}.json"

# Execute commands
hailortcli run2 -m raw_sync measure-fw-actions set-net "${HEF_PATH}"
hailo profiler "${HAR_PATH}" --runtime-data "${RUNTIME_JSON}"
