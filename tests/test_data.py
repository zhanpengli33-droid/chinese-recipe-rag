from recipe_rag.models import load_eval_queries, load_recipes
from scripts.build_demo_data import build_demo_data


def test_packaged_data_has_resume_scale() -> None:
    assert len(load_recipes()) == 323
    assert len(load_eval_queries()) == 120


def test_demo_data_generation_is_deterministic(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_demo_data(first)
    build_demo_data(second)
    for filename in ("recipes.json", "eval_queries.json", "ranking_snapshot.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_missing_data_file_has_clear_error(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    try:
        load_recipes(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
