import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

# --- Configuration ---
OUTPUT_DIR = "calib_set"
FILE_NAME = "imagenet_calib.npy"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, FILE_NAME)

NUM_SAMPLES = 32  # Hailo 권장 Calibration Batch Size (보통 32 or 64)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 경고 메시지 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_calib_data_from_tfds():
    print(f"[Info] Loading ImageNetV2 dataset (same as TFLite script)...")
    
    # 1. Load Dataset
    ds = tfds.load('imagenet_v2', split='test', as_supervised=True, shuffle_files=True)

    # 2. Preprocessing Function
    def preprocess_for_resnet(image, label):
        # Resize
        image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        
        # ResNet Preprocess (Caffe Style: BGR conversion, Mean subtraction)
        image = tf.keras.applications.resnet.preprocess_input(image)
        return image

    # 3. Apply Map & Batch
    ds = ds.map(preprocess_for_resnet, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.take(NUM_SAMPLES)

    # 4. Collect Data into Numpy Array
    print(f"[Info] Processing {NUM_SAMPLES} images...")
    data_list = []
    
    for image in ds:
        data_list.append(image.numpy())

    # Stack to create (BATCH_SIZE, 224, 224, 3)
    calib_data = np.array(data_list, dtype=np.float32)

    # 5. Save to .npy inside the directory
    # 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        print(f"[Info] Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[Info] Final Data Shape: {calib_data.shape}")
    print(f"[Info] Data Range: Min={np.min(calib_data):.2f}, Max={np.max(calib_data):.2f}")
    
    np.save(OUTPUT_PATH, calib_data)
    print(f"[Success] Saved calibration data to '{os.path.abspath(OUTPUT_PATH)}'")

if __name__ == "__main__":
    create_calib_data_from_tfds()