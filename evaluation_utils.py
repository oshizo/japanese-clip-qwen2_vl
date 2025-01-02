from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def resize_image(img, max_size):
    # 縦横の長いほうをmax_sizeに合わせて縮小
    if max(img.size) <= max_size:
        return img
    ratio = max_size / max(img.size)
    new_size = tuple(int(dim * ratio) for dim in img.size)
    return img.resize(new_size, Image.Resampling.LANCZOS)


class MultiModalInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    def __init__(
        self,
        queries: dict[str, str],  # qid => query text
        corpus: dict[str, str | Image.Image],  # cid => image path or PIL.Image
        relevant_docs: dict[str, set[str]],  # qid => Set[cid]
        **kwargs
    ) -> None:
        self.original_corpus = corpus
        super().__init__(queries, corpus, relevant_docs, **kwargs)

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs
    ) -> dict[str, float]:

        # 先にcorpusをすべてencode
        corpus_embeddings = model.encode(
            list(self.original_corpus.values()),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )

        # corpusのembeddingを渡して、通常のInformationRetrievalEvaluatorを使って評価
        return super().__call__(
            model,
            output_path=output_path,
            epoch=epoch,
            steps=steps,
            corpus_embeddings=corpus_embeddings,
            *args,
            **kwargs
        )


# colqwen2をsentence-transformersのevaluatorで評価するためのラッパー
from dataclasses import dataclass
from typing import Any, Dict, List
from tqdm.auto import tqdm
import torch


@dataclass
class ModelCardData:
    def set_evaluation_metrics(self, evaluator: Any, metrics: Dict[str, Any]) -> None:
        pass


class ColQwen2Wrapper:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.similarity_fn_name = "maxsim"
        self.model_card_data = ModelCardData()

    def encode(self, inputs, batch_size=32, show_progress_bar=False, **kwargs):
        is_query = isinstance(inputs[0], str)
        batches = range(0, len(inputs), batch_size)
        if show_progress_bar:
            batches = tqdm(
                batches,
                desc="Encoding" + (" queries" if is_query else " corpus"),
                total=(len(inputs) + batch_size - 1) // batch_size,
            )
        with torch.no_grad():
            all_embeddings = []
            for i in batches:
                batch_inputs = inputs[i : i + batch_size]

                if is_query:
                    processed = self.processor.process_queries(batch_inputs).to(
                        self.model.device
                    )
                else:
                    processed = self.processor.process_images(batch_inputs).to(
                        self.model.device
                    )

                with torch.no_grad():
                    batch_embeddings = self.model(**processed)
                    batch_embeddings = [emb.detach().cpu() for emb in batch_embeddings]
                    all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def similarity(
        self, queries_emb: List[torch.Tensor], corpus_emb: List[torch.Tensor]
    ) -> torch.Tensor:
        return self.processor.score_multi_vector(queries_emb, corpus_emb, batch_size=32)
