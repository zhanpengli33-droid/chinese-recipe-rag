from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .models import EvalQuery, load_eval_queries, load_ranking_snapshot


@dataclass(frozen=True)
class EvaluationReport:
    query_count: int
    baseline_hits: int
    hybrid_hits: int
    baseline_hit_at_5: float
    hybrid_hit_at_5: float
    relative_improvement: float


def count_hits_at_k(
    queries: list[EvalQuery], rankings: dict[str, list[str]], *, k: int = 5
) -> int:
    if k < 1:
        raise ValueError("k must be positive")
    hits = 0
    for query in queries:
        retrieved = rankings.get(query.id, ())[:k]
        if set(query.expected_recipe_ids).intersection(retrieved):
            hits += 1
    return hits


def hit_at_k(
    queries: list[EvalQuery], rankings: dict[str, list[str]], *, k: int = 5
) -> float:
    if not queries:
        raise ValueError("At least one evaluation query is required")
    return count_hits_at_k(queries, rankings, k=k) / len(queries)


def evaluate_snapshot(
    queries: list[EvalQuery] | None = None,
    snapshot: dict[str, Any] | None = None,
) -> EvaluationReport:
    queries = queries or load_eval_queries()
    snapshot = snapshot or load_ranking_snapshot()
    baseline_rankings = snapshot["baseline"]
    hybrid_rankings = snapshot["hybrid"]
    baseline_hits = count_hits_at_k(queries, baseline_rankings, k=5)
    hybrid_hits = count_hits_at_k(queries, hybrid_rankings, k=5)
    baseline_rate = baseline_hits / len(queries)
    hybrid_rate = hybrid_hits / len(queries)
    relative_improvement = (
        (hybrid_rate - baseline_rate) / baseline_rate if baseline_rate else 0.0
    )
    return EvaluationReport(
        query_count=len(queries),
        baseline_hits=baseline_hits,
        hybrid_hits=hybrid_hits,
        baseline_hit_at_5=baseline_rate,
        hybrid_hit_at_5=hybrid_rate,
        relative_improvement=relative_improvement,
    )


def main() -> None:
    print(json.dumps(asdict(evaluate_snapshot()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
