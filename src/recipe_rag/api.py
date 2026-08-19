from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .models import PipelineResult, load_recipes
from .pipeline import RecipeRAGPipeline
from .retriever import HybridRetriever


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


def _validate_request(request: QueryRequest) -> tuple[str, int]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if len(query) > 200:
        raise HTTPException(status_code=400, detail="query cannot exceed 200 characters")
    if not 1 <= request.top_k <= 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
    return query, request.top_k


def _result_payload(result: PipelineResult) -> dict[str, object]:
    return {
        "query": result.original_query,
        "rewritten_query": result.rewritten_query,
        "query_type": result.query_type,
        "results": [hit.to_dict() for hit in result.hits],
        "answer": result.answer,
    }


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(pipeline: RecipeRAGPipeline | None = None) -> FastAPI:
    pipeline = pipeline or RecipeRAGPipeline(HybridRetriever(load_recipes()))
    application = FastAPI(
        title="Chinese Recipe RAG System",
        version="0.1.0",
        description="Offline hybrid retrieval and streaming recipe Q&A.",
    )

    @application.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "recipe_count": len(pipeline.retriever.recipes)}

    @application.post("/search")
    def search(request: QueryRequest) -> dict[str, object]:
        query, top_k = _validate_request(request)
        return _result_payload(pipeline.run(query, top_k=top_k))

    @application.post("/chat/stream")
    def stream_chat(request: QueryRequest) -> StreamingResponse:
        query, top_k = _validate_request(request)

        def events() -> Iterator[str]:
            try:
                result = pipeline.run(query, top_k=top_k)
                yield _sse(
                    "route",
                    {
                        "query_type": result.query_type,
                        "rewritten_query": result.rewritten_query,
                    },
                )
                yield _sse(
                    "retrieval",
                    {"results": [hit.to_dict() for hit in result.hits]},
                )
                yield _sse("answer", {"answer": result.answer})
            except Exception as exc:  # The stream must terminate with a valid SSE event.
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return application


app = create_app()
