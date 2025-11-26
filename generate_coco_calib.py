import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

# --- Configuration ---
OUTPUT_DIR = "calib_set"
FILE_NAME = "yolov4_calib.npy"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, FILE_NAME)

NUM_SAMPLES = 32  # Hailo recommended Calibration Batch Size
# Updated based on check_onnx_input_shape.py result
IMG_HEIGHT = 416 
IMG_WIDTH = 416

# Suppress TensorFlow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_calib_data_from_coco():
    print(f"[Info] Loading COCO 2017 Validation dataset...")
    
    # 1. Load Dataset (COCO 2017)
    # as_supervised=False returns a dictionary containing 'image', 'label', etc.
    ds = tfds.load('coco/2017', split='validation', shuffle_files=True)

    # 2. Preprocessing Function for YOLOv4
    def preprocess_for_yolo(features):
        image = features['image'] # Extract image from dictionary

        # Resize (Bilinear interpolation)
        image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        
        # Normalize [0, 255] -> [0.0, 1.0]
        # Validated: Input is NHWC [416, 416, 3] based on ONNX check
        image = tf.cast(image, tf.float32) / 255.0
        
        return image

    # 3. Apply Map & Batch
    ds = ds.map(preprocess_for_yolo, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.take(NUM_SAMPLES)

    # 4. Collect Data into Numpy Array
    print(f"[Info] Processing {NUM_SAMPLES} images...")
    data_list = []
    
    for image in ds:
        data_list.append(image.numpy())

    # Stack to create (BATCH_SIZE, 416, 416, 3)
    calib_data = np.array(data_list, dtype=np.float32)

    # 5. Save to .npy inside the directory
    if not os.path.exists(OUTPUT_DIR):
        print(f"[Info] Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[Info] Final Data Shape: {calib_data.shape}")
    print(f"[Info] Data Range: Min={np.min(calib_data):.2f}, Max={np.max(calib_data):.2f}")
    
    np.save(OUTPUT_PATH, calib_data)
    print(f"[Success] Saved calibration data to '{os.path.abspath(OUTPUT_PATH)}'")

if __name__ == "__main__":
    create_calib_data_from_coco()