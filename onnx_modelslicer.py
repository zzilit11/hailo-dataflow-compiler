import onnx
from onnx import utils, shape_inference
import os

def split_onnx_model(input_model_path, split_tensor_name, output_part1, output_part2):
    """
    Splits an ONNX model into two parts at the specified tensor.

    Args:
        input_model_path (str): Path to the original ONNX model.
        split_tensor_name (str): The name of the tensor where the split occurs.
                                 (Output of Part 1, Input of Part 2)
        output_part1 (str): Path to save the first part.
        output_part2 (str): Path to save the second part.
    """
    
    # 1. Load Model and Validation
    print(f"[Info] Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    onnx.checker.check_model(model)

    # 2. Shape Inference (Critical Step)
    # 분할 지점의 Tensor Shape 정보를 확정하기 위해 필수적이다.
    # 이를 수행하지 않으면 Part 2의 Input Shape 정보를 잃을 수 있다.
    print("[Info] Applying shape inference...")
    inferred_model = shape_inference.infer_shapes(model)
    
    # 추출을 위해 shape info가 포함된 임시 모델 저장
    temp_inferred_path = "temp_inferred_model.onnx"
    onnx.save(inferred_model, temp_inferred_path)

    try:
        # 3. Get Original Input/Output Names
        # shape inference된 모델 기준
        original_inputs = [node.name for node in inferred_model.graph.input]
        original_outputs = [node.name for node in inferred_model.graph.output]

        print(f"[Info] Split Tensor: {split_tensor_name}")

        # 4. Extract Part 1
        # Inputs: Original Inputs
        # Outputs: Split Tensor
        print(f"[Info] Extracting Part 1 -> {output_part1}")
        utils.extract_model(
            temp_inferred_path,
            output_part1,
            input_names=original_inputs,
            output_names=[split_tensor_name]
        )

        # 5. Extract Part 2
        # Inputs: Split Tensor
        # Outputs: Original Outputs
        print(f"[Info] Extracting Part 2 -> {output_part2}")
        utils.extract_model(
            temp_inferred_path,
            output_part2,
            input_names=[split_tensor_name],
            output_names=original_outputs
        )

        # 6. Verify Created Models
        print("[Info] Verifying split models...")
        onnx.checker.check_model(output_part1)
        onnx.checker.check_model(output_part2)
        print("[Success] Partitioning complete.")

    except Exception as e:
        print(f"[Error] Failed to split model: {e}")
        raise e
    finally:
        if os.path.exists(temp_inferred_path):
            os.remove(temp_inferred_path)

# --- Usage Example ---
if __name__ == "__main__":
    # 설정 예시
    MODEL_PATH = "my_model.onnx"       # 원본 모델 경로
    SPLIT_NODE = "conv1_output_tensor" # 분할할 텐서 이름 (Netron 등으로 확인 필요)
    PART1_PATH = "part1.onnx"
    PART2_PATH = "part2.onnx"

    # 실제 사용 시 아래 주석 해제 및 경로 수정
    # split_onnx_model(MODEL_PATH, SPLIT_NODE, PART1_PATH, PART2_PATH)