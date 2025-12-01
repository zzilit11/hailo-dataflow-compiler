#!/bin/bash

mkdir -p traces

TRACE_PATH=$1
if [ -z "$TRACE_PATH" ]; then
    echo "Please provide a trace file path. Example: ./run_tracer.sh <trace_file>.hrtt"
    exit 1
fi

# Extract base name without extension
BASENAME=$(basename "$TRACE_PATH" .hrtt)

# Execute command
hailo runtime-profiler "${TRACE_PATH}"

# Output file from profiler
mv runtime_report.html "traces/${BASENAME}_runtime_report.html"
