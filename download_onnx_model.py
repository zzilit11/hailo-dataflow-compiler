import os
import urllib.request
import sys

# Configuration
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet152-v1-7.onnx"
#MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
#MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov4/model/yolov4.onnx"
DOWNLOAD_DIR = "onnx_models"
#FILE_NAME = "resnet50_v1.onnx"
#FILE_NAME = "mobilenetv2.onnx"
FILE_NAME = "resnet152_v1.onnx"
OUTPUT_PATH = os.path.join(DOWNLOAD_DIR, FILE_NAME)

def download_model(url, output_path):
    # Check and create directory
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        print(f"[Info] Creating directory: '{directory}'")
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(output_path):
        print(f"[Info] File '{output_path}' already exists. Skipping download.")
        return

    print(f"[Info] Starting download: {output_path}")
    print(f"[Source] {url}")
    
    try:
        # Progress hook function
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded * 100 / total_size
                sys.stdout.write(f"\r[Download] Progress: {percent:.1f}% ({downloaded/1024/1024:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print("\n[Success] Download complete.")
        
    except Exception as e:
        print(f"\n[Error] Download failed: {e}")
        # Remove incomplete file on failure
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    download_model(MODEL_URL, OUTPUT_PATH)