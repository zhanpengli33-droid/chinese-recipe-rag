from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

from .models import Recipe, SearchHit


_ASCII_WORD = re.compile(r"[a-z0-9]+")
_CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese text with character unigrams/bigrams plus ASCII words."""
    normalized = text.lower().strip()
    chinese = _CHINESE_CHAR.findall(normalized)
    bigrams = ["".join(chinese[index : index + 2]) for index in range(len(chinese) - 1)]
    return chinese + bigrams + _ASCII_WORD.findall(normalized)


def _hash_embedding(text: str, dimensions: int = 256) -> tuple[float, ...]:
    vector = [0.0] * dimensions
    for token, count in Counter(tokenize(text)).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimensions
        sign = 1.0 if value & (1 << 63) else -1.0
        vector[index] += sign * float(count)
    norm = math.sqrt(sum(component * component for component in vector))
    if norm:
        vector = [component / norm for component in vector]
    return tuple(vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, rank_constant: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    """Deterministic BM25 + dense retrieval with RRF fusion."""

    def __init__(self, recipes: Sequence[Recipe], *, dimensions: int = 256) -> None:
        if not recipes:
            raise ValueError("At least one recipe is required")
        self.recipes = tuple(recipes)
        self._by_id = {recipe.id: recipe for recipe in self.recipes}
        if len(self._by_id) != len(self.recipes):
            raise ValueError("Recipe IDs must be unique")

        self._documents = [tokenize(recipe.document_text) for recipe in self.recipes]
        self._frequencies = [Counter(tokens) for tokens in self._documents]
        self._average_length = sum(map(len, self._documents)) / len(self._documents)
        self._document_frequency = Counter(
            token for frequencies in self._frequencies for token in frequencies
        )
        self._dense_vectors = [
            _hash_embedding(recipe.document_text, dimensions) for recipe in self.recipes
        ]
        self._dimensions = dimensions

    def _bm25_scores(self, query: str) -> dict[str, float]:
        query_tokens = tokenize(query)
        document_count = len(self.recipes)
        k1 = 1.5
        b = 0.75
        scores: dict[str, float] = {}
        for recipe, frequencies, tokens in zip(
            self.recipes, self._frequencies, self._documents, strict=True
        ):
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[token]
                inverse_document_frequency = math.log(
                    1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0 - b + b * len(tokens) / self._average_length
                )
                score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
            scores[recipe.id] = score
        return scores

    def _dense_scores(self, query: str) -> dict[str, float]:
        query_vector = _hash_embedding(query, self._dimensions)
        return {
            recipe.id: _dot(query_vector, vector)
            for recipe, vector in zip(self.recipes, self._dense_vectors, strict=True)
        }

    @staticmethod
    def _ranking(scores: dict[str, float]) -> list[str]:
        return [item_id for item_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]

    def search(self, query: str, top_k: int = 5, *, mode: str = "hybrid") -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        lexical_scores = self._bm25_scores(query)
        dense_scores = self._dense_scores(query)
        if mode == "lexical":
            fused = [(item_id, lexical_scores[item_id]) for item_id in self._ranking(lexical_scores)]
        elif mode == "dense":
            fused = [(item_id, dense_scores[item_id]) for item_id in self._ranking(dense_scores)]
        elif mode == "hybrid":
            fused = reciprocal_rank_fusion(
                (self._ranking(lexical_scores), self._ranking(dense_scores))
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        return [
            SearchHit(
                recipe=self._by_id[item_id],
                score=score,
                lexical_score=lexical_scores[item_id],
                dense_score=dense_scores[item_id],
            )
            for item_id, score in fused[: min(top_k, len(fused))]
        ]


class BGEFaissRetriever:
    """Optional BGE + FAISS backend; install the ``full`` dependency group."""

    def __init__(
        self,
        recipes: Sequence[Recipe],
        *,
        model_name: str = "BAAI/bge-small-zh-v1.5",
    ) -> None:
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                'BGE/FAISS dependencies are unavailable; install with pip install -e ".[full]"'
            ) from exc

        self._recipes = tuple(recipes)
        self._np = np
        self._model = SentenceTransformer(model_name)
        documents = [recipe.document_text for recipe in self._recipes]
        embeddings = self._model.encode(
            documents, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")
        vector = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = self._index.search(vector, min(top_k, len(self._recipes)))
        return [
            SearchHit(recipe=self._recipes[index], score=float(score), dense_score=float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
        ]
