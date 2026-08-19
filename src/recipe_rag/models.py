from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    cuisine: str
    ingredients: tuple[str, ...]
    steps: tuple[str, ...]
    description: str
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Recipe":
        return cls(
            id=value["id"],
            title=value["title"],
            cuisine=value["cuisine"],
            ingredients=tuple(value["ingredients"]),
            steps=tuple(value["steps"]),
            description=value["description"],
            tags=tuple(value.get("tags", ())),
        )

    @property
    def document_text(self) -> str:
        return " ".join(
            (
                self.title,
                self.cuisine,
                self.description,
                " ".join(self.ingredients),
                " ".join(self.steps),
                " ".join(self.tags),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalQuery:
    id: str
    query: str
    query_type: str
    expected_recipe_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalQuery":
        return cls(
            id=value["id"],
            query=value["query"],
            query_type=value["query_type"],
            expected_recipe_ids=tuple(value["expected_recipe_ids"]),
        )


@dataclass(frozen=True)
class SearchHit:
    recipe: Recipe
    score: float
    lexical_score: float = 0.0
    dense_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe.id,
            "title": self.recipe.title,
            "cuisine": self.recipe.cuisine,
            "ingredients": list(self.recipe.ingredients),
            "score": round(self.score, 6),
            "lexical_score": round(self.lexical_score, 6),
            "dense_score": round(self.dense_score, 6),
        }


@dataclass(frozen=True)
class PipelineResult:
    original_query: str
    rewritten_query: str
    query_type: str
    hits: tuple[SearchHit, ...]
    answer: str


def _load_json(filename: str, path: str | Path | None = None) -> Any:
    if path is not None:
        source = Path(path)
        try:
            return json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Data file not found: {source}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in data file {source}: {exc.msg}") from exc

    source = resources.files("recipe_rag.data").joinpath(filename)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Packaged data file not found: {filename}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in packaged data file {filename}: {exc.msg}") from exc


def load_recipes(path: str | Path | None = None) -> list[Recipe]:
    values = _load_json("recipes.json", path)
    return [Recipe.from_dict(value) for value in values]


def load_eval_queries(path: str | Path | None = None) -> list[EvalQuery]:
    values = _load_json("eval_queries.json", path)
    return [EvalQuery.from_dict(value) for value in values]


def load_ranking_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    return _load_json("ranking_snapshot.json", path)
