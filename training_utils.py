from itertools import accumulate
import torch
from PIL import Image
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss


def resize_image(img, max_size):
    # 縦横の長いほうをmax_sizeに合わせて縮小
    if max(img.size) <= max_size:
        return img
    ratio = max_size / max(img.size)
    new_size = tuple(int(dim * ratio) for dim in img.size)
    return img.resize(new_size, Image.Resampling.LANCZOS)


class ImageTextCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    def embed_minibatch_iter(
        self, sentence_feature, with_grad, copy_random_state, random_states=None
    ):
        # 画像データの場合
        if "image_grid_thw" in sentence_feature:

            patches_per_image = [
                (t * h * w).item() for t, h, w in sentence_feature["image_grid_thw"]
            ]
            cumsum_patches = [0] + list(accumulate(patches_per_image))

            for i in range(0, len(patches_per_image), self.mini_batch_size):
                batch_indices = []
                batch_grid_thw = []

                for j in range(
                    i, min(i + self.mini_batch_size, len(patches_per_image))
                ):
                    start_idx = cumsum_patches[j]
                    end_idx = cumsum_patches[j + 1]
                    batch_indices.extend(range(start_idx, end_idx))
                    batch_grid_thw.append(sentence_feature["image_grid_thw"][j])

                batch_feature = {
                    "pixel_values": sentence_feature["pixel_values"][batch_indices],
                    "image_grid_thw": torch.stack(batch_grid_thw),
                    "image_text_info": sentence_feature["image_text_info"][
                        i : i + self.mini_batch_size
                    ],
                }

                batch_idx = i // self.mini_batch_size
                random_state = (
                    random_states[batch_idx] if random_states is not None else None
                )
                yield self.embed_minibatch(
                    sentence_feature=batch_feature,
                    begin=0,
                    end=None,
                    with_grad=with_grad,
                    copy_random_state=copy_random_state,
                    random_state=random_state,
                )

        # テキストデータの場合（既存の実装）
        else:
            for i, b in enumerate(
                range(
                    0, len(next(iter(sentence_feature.values()))), self.mini_batch_size
                )
            ):
                e = b + self.mini_batch_size
                yield self.embed_minibatch(
                    sentence_feature=sentence_feature,
                    begin=b,
                    end=e,
                    with_grad=with_grad,
                    copy_random_state=copy_random_state,
                    random_state=(
                        random_states[i] if random_states is not None else None
                    ),
                )
