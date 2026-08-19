import pytest

from recipe_rag.pipeline import QueryType, RecipeRAGPipeline, rewrite_query, route_query
from recipe_rag.retriever import HybridRetriever


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("推荐一道牛肉菜", QueryType.RECOMMENDATION),
        ("红烧牛腩需要什么食材", QueryType.INGREDIENTS),
        ("红烧牛腩怎么做", QueryType.STEPS),
    ],
)
def test_query_routing(query, expected) -> None:
    assert route_query(query) is expected


def test_query_rewriter_removes_fillers() -> None:
    assert rewrite_query("请问麻烦告诉我红烧牛腩怎么做一下？") == "告诉我红烧牛腩怎么做"


def test_pipeline_answer_is_grounded_in_recipe(sample_recipes) -> None:
    pipeline = RecipeRAGPipeline(HybridRetriever(sample_recipes))
    result = pipeline.run("红烧牛腩需要什么食材", top_k=2)
    assert result.query_type == "ingredients"
    assert "牛腩" in result.answer
    assert "recipe-b" in result.answer
