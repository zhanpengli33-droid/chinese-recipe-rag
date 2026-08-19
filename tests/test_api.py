from fastapi.testclient import TestClient

from recipe_rag.api import create_app
from recipe_rag.pipeline import RecipeRAGPipeline
from recipe_rag.retriever import HybridRetriever


def _client(sample_recipes) -> TestClient:
    pipeline = RecipeRAGPipeline(HybridRetriever(sample_recipes))
    return TestClient(create_app(pipeline))


def test_health_and_search_endpoints(sample_recipes) -> None:
    client = _client(sample_recipes)
    assert client.get("/health").json() == {"status": "ok", "recipe_count": 3}
    response = client.post("/search", json={"query": "红烧牛腩怎么做", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["results"][0]["recipe_id"] == "recipe-b"


def test_search_rejects_empty_query(sample_recipes) -> None:
    response = _client(sample_recipes).post("/search", json={"query": "   "})
    assert response.status_code == 400


def test_sse_stream_has_expected_event_order(sample_recipes) -> None:
    response = _client(sample_recipes).post(
        "/chat/stream", json={"query": "红烧牛腩怎么做", "top_k": 2}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line for line in response.text.splitlines() if line.startswith("event:")]
    assert events == ["event: route", "event: retrieval", "event: answer"]
