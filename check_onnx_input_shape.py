import onnx
import sys
import os

# 확인하고 싶은 모델 경로
MODEL_PATH = "onnx_models/efficientvit_b0.onnx"

def check_model_input(path):
    # 1. Check if file exists
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return

    print(f"[Info] Loading model: {path}...")
    
    # 2. Load ONNX model
    try:
        model = onnx.load(path)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    # 3. Inspect Inputs
    print("\n" + "="*40)
    print("      Model Input Configuration")
    print("="*40)

    for i, input_tensor in enumerate(model.graph.input):
        name = input_tensor.name
        
        # Get Shape
        tensor_type = input_tensor.type.tensor_type
        dims = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    # Fixed dimension (e.g., 608)
                    dims.append(str(d.dim_value))
                elif d.HasField("dim_param"):
                    # Dynamic dimension (e.g., "batch_size")
                    dims.append(f"Dynamic({d.dim_param})")
                else:
                    dims.append("?")
        
        shape_str = "x".join(dims)
        print(f"Input #{i}: '{name}'")
        print(f" - Shape: [{shape_str}]")
        print("-" * 40)

        # Check specifically for YOLO resolution
        if len(dims) == 4:
            # Assuming NCHW or NHWC, Height/Width are usually the larger numbers or at index 2,3
            print(f"[Result] Check these dimensions for resolution: {dims}")

if __name__ == "__main__":
    # Command line argument support: python check_onnx_shape.py <path_to_model>
    target_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    check_model_input(target_path)