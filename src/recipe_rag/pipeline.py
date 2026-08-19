from __future__ import annotations

from enum import StrEnum

from .models import PipelineResult, SearchHit
from .retriever import HybridRetriever


class QueryType(StrEnum):
    RECOMMENDATION = "recommendation"
    INGREDIENTS = "ingredients"
    STEPS = "steps"


_STEP_TERMS = ("怎么做", "如何做", "做法", "步骤", "教我做")
_INGREDIENT_TERMS = ("食材", "原料", "需要什么", "放什么", "用什么")
_FILLER_TERMS = ("请问", "我想知道", "能不能", "麻烦", "一下", "呀", "呢")


def route_query(query: str) -> QueryType:
    if any(term in query for term in _STEP_TERMS):
        return QueryType.STEPS
    if any(term in query for term in _INGREDIENT_TERMS):
        return QueryType.INGREDIENTS
    return QueryType.RECOMMENDATION


def rewrite_query(query: str) -> str:
    rewritten = query.strip()
    for term in _FILLER_TERMS:
        rewritten = rewritten.replace(term, "")
    return " ".join(rewritten.split()).strip(" ，,?？") or query.strip()


def _render_answer(query_type: QueryType, hits: list[SearchHit]) -> str:
    if not hits:
        return "暂未检索到可靠的食谱证据，建议补充菜名或主要食材后重试。"

    if query_type is QueryType.INGREDIENTS:
        recipe = hits[0].recipe
        return (
            f"《{recipe.title}》主要食材："
            f"{'、'.join(recipe.ingredients)}。来源：{recipe.id}。"
        )
    if query_type is QueryType.STEPS:
        recipe = hits[0].recipe
        steps = " ".join(f"{index}. {step}" for index, step in enumerate(recipe.steps, start=1))
        return f"《{recipe.title}》做法：{steps} 来源：{recipe.id}。"

    recommendations = "；".join(
        f"《{hit.recipe.title}》（{hit.recipe.cuisine}，来源 {hit.recipe.id}）"
        for hit in hits[:3]
    )
    return f"根据检索结果，可以尝试：{recommendations}。"


class RecipeRAGPipeline:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def run(self, query: str, *, top_k: int = 5) -> PipelineResult:
        original_query = query.strip()
        if not original_query:
            raise ValueError("Query cannot be empty")
        query_type = route_query(original_query)
        rewritten_query = rewrite_query(original_query)
        hits = self.retriever.search(rewritten_query, top_k=top_k)
        return PipelineResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            query_type=query_type.value,
            hits=tuple(hits),
            answer=_render_answer(query_type, hits),
        )
