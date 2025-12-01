import os
import torch
import timm

def main():
    # 저장 디렉토리 생성
    save_dir = "onnx_models"
    os.makedirs(save_dir, exist_ok=True)

    # 1. EfficientViT-B0 모델 로드 (ImageNet-1k pretrained)
    model_name = "efficientvit_b0.r224_in1k"
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    # 2. 더미 입력 (batch=1, 3x224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. ONNX export 경로
    onnx_path = os.path.join(save_dir, "efficientvit_b0.onnx")

    # 4. ONNX export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"Export 완료: {onnx_path}")

if __name__ == "__main__":
    main()
