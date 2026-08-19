from recipe_rag.evaluation import evaluate_snapshot, hit_at_k
from recipe_rag.models import EvalQuery


def test_hit_at_k_uses_expected_recipe_ids() -> None:
    queries = [
        EvalQuery("q1", "query one", "steps", ("r1",)),
        EvalQuery("q2", "query two", "ingredients", ("r2",)),
    ]
    rankings = {"q1": ["x", "r1"], "q2": ["x", "y"]}
    assert hit_at_k(queries, rankings, k=2) == 0.5


def test_packaged_snapshot_reproduces_resume_metrics() -> None:
    report = evaluate_snapshot()
    assert report.query_count == 120
    assert report.baseline_hits == 82
    assert report.hybrid_hits == 106
    assert round(report.baseline_hit_at_5 * 100, 1) == 68.3
    assert round(report.hybrid_hit_at_5 * 100, 1) == 88.3
    assert round(report.relative_improvement * 100, 1) == 29.3
