# coding=utf-8
# Copyright 2024 oshizo
#
# This implementation is based on:
# 1. Qwen2-VL (https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/)
#    Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#    Originally based on EleutherAI's GPT-NeoX library and GPT-NeoX/OPT implementations.
#
# 2. CLIP (https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/)
#    Copyright 2021 The OpenAI Team Authors and The HuggingFace Team.
#    CLIP Configuration
#    Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLIPQwen2VL model implementation."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch import nn
from transformers import BertConfig, BertModel, PretrainedConfig, PreTrainedModel
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)


class CLIPQwen2VLConfig(PretrainedConfig):
    model_type = "clip_qwen2vl"

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        projection_dim: int = 1024,
        logit_scale_init_value: float = 2.6592,
        **kwargs,
    ):
        super().__init__(**kwargs)

        text_config = text_config or {}
        vision_config = vision_config or {}

        self.text_config = BertConfig(**text_config)
        self.vision_config = Qwen2VLVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value


class CLIPQwen2VLModel(PreTrainedModel):
    config_class = CLIPQwen2VLConfig

    def __init__(self, config: CLIPQwen2VLConfig):
        super().__init__(config)

        self.projection_dim = config.text_config.hidden_size  # 1024
        self.text_embed_dim = config.text_config.hidden_size  # 1024
        self.vision_embed_dim = config.vision_config.hidden_size  # 1536

        # Text encoder
        self.text_model = BertModel(config.text_config)

        # Vision encoder
        self.vision_model = Qwen2VisionTransformerPretrainedModel(config.vision_config)

        # vision projection (1536 -> 1024)
        self.vision_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Mean pooling
        attention_mask = attention_mask.to(text_outputs.last_hidden_state.dtype)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            text_outputs.last_hidden_state.size()
        )
        sum_embeddings = torch.sum(
            text_outputs.last_hidden_state * input_mask_expanded, 1
        )
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        text_embeds = sum_embeddings / sum_mask

        return text_embeds

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        batch_size = image_grid_thw.shape[0]
        spatial_merge_size = 2

        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        vision_output = self.vision_model(
            hidden_states=pixel_values, grid_thw=image_grid_thw
        )

        merged_patches_per_image = [
            ((h // spatial_merge_size) * (w // spatial_merge_size) * t).item()
            for t, h, w in image_grid_thw
        ]
        merged_cu_seqlens = torch.tensor(
            [0] + list(itertools.accumulate(merged_patches_per_image)),
            device=vision_output.device,
        )

        image_features = []
        for i in range(batch_size):
            start_idx = merged_cu_seqlens[i]
            end_idx = merged_cu_seqlens[i + 1]
            image_features.append(vision_output[start_idx:end_idx].mean(dim=0))

        image_features = torch.stack(image_features)
        image_embeds = self.vision_projection(image_features)
        return image_embeds


class CLIPQwen2VLWrapper(nn.Module):
    save_in_root: bool = True

    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: str = None,
        backend: str = "torch",
        enable_text_grad: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.enable_text_grad = enable_text_grad

        model_args = kwargs.get("model_args", {})
        if "torch_dtype" not in model_args:
            model_args["torch_dtype"] = torch.bfloat16

        self.model = CLIPQwen2VLModel.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, **model_args
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "cl-nagoya/ruri-large"
        )
        self.processor = transformers.AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct"
        )

    def __repr__(self) -> str:
        return "CLIPQwen2VLWrapper()"

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_embeds = []
        text_embeds = []

        if "pixel_values" in features:
            image_embeds = self.model.get_image_features(
                pixel_values=features["pixel_values"],
                image_grid_thw=features["image_grid_thw"],
            )

        if "input_ids" in features:
            text_embeds = self.model.get_text_features(
                input_ids=features["input_ids"],
                attention_mask=features.get("attention_mask", None),
                position_ids=features.get("position_ids", None),
                output_attentions=features.get("output_attentions", None),
                output_hidden_states=features.get("output_hidden_states", None),
            )
            if self.enable_text_grad:
                # peftでの学習時にtext modelの層を指定しない場合のエラー回避
                text_embeds = text_embeds.detach().requires_grad_()

        sentence_embedding = []
        image_features = iter(image_embeds)
        text_features = iter(text_embeds)

        for idx, input_type in enumerate(features["image_text_info"]):
            if input_type == 0:
                sentence_embedding.append(next(image_features))
            else:
                sentence_embedding.append(next(text_features))

        features["sentence_embedding"] = torch.stack(sentence_embedding).float()

        return features

    def tokenize(
        self, texts: List[Union[str, Image.Image]], padding: str | bool = True
    ) -> dict[str, torch.Tensor]:
        images = []
        texts_values = []
        image_text_info = []

        for idx, data in enumerate(texts):
            if isinstance(data, Image.Image):
                images.append(data)
                image_text_info.append(0)
            else:
                texts_values.append(data)
                image_text_info.append(1)

        encoding = {}
        if len(texts_values):
            encoding = self.tokenizer(
                texts_values,
                return_tensors="pt",
                padding=padding,
                truncation=True,
                max_length=512,
            )

        if len(images):
            image_features = self.processor.image_processor(images, return_tensors="pt")
            encoding.update(image_features)

        encoding["image_text_info"] = image_text_info
        return dict(encoding)

    @property
    def processor(self) -> transformers.PreTrainedModel:
        return self._processor

    @processor.setter
    def processor(self, processor):
        self._processor = processor

    def save(self, output_path: str) -> None:
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    @staticmethod
    def load(input_path: str) -> CLIPQwen2VLWrapper:
        return CLIPQwen2VLWrapper(model_name_or_path=input_path)
