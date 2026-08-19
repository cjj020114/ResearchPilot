from __future__ import annotations

from pathlib import Path
from typing import Sequence

from backend.app.indexing.embeddings import EmbeddingProvider, HashEmbeddingProvider


class ClipEmbeddingProvider(EmbeddingProvider):
    """CLIP text+image encoders in a shared vector space (方案 1A).

    Text: multilingual CLIP text tower.
    Image: original CLIP ViT image tower (aligned with the text tower).
    """

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        image_model_name: str = "clip-ViT-B-32",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.text_model = SentenceTransformer(text_model_name)
        self.image_model = SentenceTransformer(image_model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.text_model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_images(self, image_paths: Sequence[str | Path]) -> list[list[float]]:
        if not image_paths:
            return []
        from PIL import Image

        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        vectors = self.image_model.encode(images, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


def build_clip_embedding_provider(
    text_model_name: str,
    image_model_name: str,
    offline: bool = False,
) -> EmbeddingProvider:
    if offline:
        return HashEmbeddingProvider()
    try:
        return ClipEmbeddingProvider(text_model_name, image_model_name)
    except Exception:
        return HashEmbeddingProvider()
