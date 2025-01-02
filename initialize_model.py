from transformers import AutoModel, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
from modeling_clip_qwen2vl import (
    CLIPQwen2VLConfig,
    CLIPQwen2VLModel,
    CLIPQwen2VLWrapper,
)

from torch import nn
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP model")
    parser.add_argument(
        "--save_path", type=str, default="./clip", help="Path to save the model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Vision Model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="cuda:0"
    )

    # 初期化
    vision = Qwen2VisionTransformerPretrainedModel(model.config.vision_config)
    vision_config = vision.config.to_dict()

    # ViTの重みをコピー
    vision_state_dict = {}
    for key, value in model.state_dict().items():
        if key.startswith("visual."):
            new_key = key.replace("visual.", "")
            vision_state_dict[new_key] = value
    vision.load_state_dict(vision_state_dict)

    # 2. Text Model
    text_model = AutoModel.from_pretrained("cl-nagoya/ruri-large")
    text_config = text_model.config.to_dict()

    # 3. CLIPの作成
    config = CLIPQwen2VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        projection_dim=1024,  # vision modelの出力次元
        logit_scale_init_value=2.6592,
    )
    clip = CLIPQwen2VLModel(config)

    # 重みをコピー
    text_state_dict = text_model.state_dict()
    for key, value in text_state_dict.items():
        clip.text_model.state_dict()[key].copy_(value)

    vision_state_dict = vision.state_dict()
    for key, value in vision_state_dict.items():
        clip.vision_model.state_dict()[key].copy_(value)

    # Projection層の初期化
    nn.init.xavier_uniform_(clip.vision_projection.weight)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    clip.save_pretrained(save_path)


if __name__ == "__main__":
    main()
