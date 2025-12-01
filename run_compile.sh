#!/bin/bash
set -e  # 에러 발생 시 스크립트 즉시 중단

# --- 설정 (파일명 등) ---
MODEL_NAME="densenet121"

BASE_DIR="$(pwd)"
ONNX_FILE="${BASE_DIR}/onnx_models/${MODEL_NAME}.onnx"

# Parser 결과물
TEMP_HAR_FILE="${MODEL_NAME}.har"
DEST_HAR_FILE="${BASE_DIR}/har_models/${MODEL_NAME}.har"

QUANTIZED_HAR_FILE="${BASE_DIR}/har_models/${MODEL_NAME}_quantized.har"
CALIB_DATA="${BASE_DIR}/calib_set/imagenet_calib.npy"
CALIB_SCRIPT="generate_imagenet_calib.py"

# --- 0. 디렉토리 생성 ---
if [ -d "onnx_models" ] && [ -d "har_models" ] && [ -d "calib_set" ] && [ -d "hef_models" ]; then
    echo "[Step 0] Directories already exist. Skipping creation."
else
    echo "[Step 0] Creating directories..."
    mkdir -p onnx_models har_models calib_set hef_models
fi

# --- [Check] Protobuf Version Check (Step 0과 1 사이) ---
echo "[Check] Verifying protobuf version..."
TARGET_PROTO="3.20.3"
CURRENT_PROTO=$(pip show protobuf | grep Version | awk '{print $2}')

if [ "$CURRENT_PROTO" != "$TARGET_PROTO" ]; then
    echo "  >> Current protobuf ($CURRENT_PROTO) does not match target ($TARGET_PROTO)."
    echo "  >> Installing protobuf==$TARGET_PROTO..."
    pip install protobuf==$TARGET_PROTO > /dev/null 2>&1
else
    echo "  >> Protobuf version is correct ($CURRENT_PROTO)."
fi

# --- 1. ONNX 모델 다운로드 ---
echo "[Step 1] Downloading ONNX model..."
python3 download_onnx_model.py

# --- 2. Hailo Parser (ONNX -> HAR) ---
echo "[Step 2] Parsing ONNX to HAR (Auto-mode)..."
yes | hailo parser onnx "${ONNX_FILE}"

# 생성된 파일을 har_models 디렉토리로 이동
echo "  >> Moving generated HAR file to har_models directory..."
if [ -f "${TEMP_HAR_FILE}" ]; then
    mv "${TEMP_HAR_FILE}" "${DEST_HAR_FILE}"
else
    echo "[Warning] Temporary HAR file not found. Check if Parser succeeded."
fi

# --- 3. Calibration 데이터 생성 ---
echo "[Step 3] Checking Calibration Data..."

if [ -f "${CALIB_DATA}" ]; then
    echo "  >> '${CALIB_DATA}' already exists. Skipping generation."
else
    echo "  >> Calibration data not found. Generating COCO Calibration Data..."

    echo "  >> Installing protobuf==4.21.12 for TensorFlow..."
    pip install protobuf==4.21.12 > /dev/null 2>&1

    echo "  >> Running generation script..."
    python3 "${CALIB_SCRIPT}"

    echo "  >> Reverting protobuf to 3.20.3 for Hailo..."
    pip install protobuf==3.20.3 > /dev/null 2>&1
fi

# --- 4. Hailo Optimize (Quantization) ---
echo "[Step 4] Optimizing (Quantizing) Model..."
hailo optimize "${DEST_HAR_FILE}" \
    --calib-set-path "${CALIB_DATA}" \
    --output-har-path "${QUANTIZED_HAR_FILE}"

# --- 5. Hailo Compiler (HAR -> HEF) ---
echo "[Step 5] Compiling to HEF..."
hailo compiler "${QUANTIZED_HAR_FILE}" --output-dir "${BASE_DIR}/hef_models"

# 생성된 _compiled.har 파일을 har_models 디렉토리로 이동
echo "  >> Moving compiled HAR file to har_models directory..."
COMPILED_HAR_SRC="${BASE_DIR}/hef_models/${MODEL_NAME}_compiled.har"
COMPILED_HAR_DST="${BASE_DIR}/har_models/"

if [ -f "${COMPILED_HAR_SRC}" ]; then
    mv "${COMPILED_HAR_SRC}" "${COMPILED_HAR_DST}"
    echo "[Info] Moved compiled HAR to ${COMPILED_HAR_DST}"
else
    echo "[Warning] Compiled HAR file not found in hef_models/"
fi

echo "------------------------------------------------"
echo "[Success] All steps completed."
echo "Final HEF file is located in 'hef_models/' directory."
echo "Compiled HAR file is located in 'har_models/' directory."