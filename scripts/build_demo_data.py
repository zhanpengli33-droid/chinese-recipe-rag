from __future__ import annotations

import json
from pathlib import Path


INGREDIENTS = [
    "鸡胸肉",
    "五花肉",
    "牛腩",
    "排骨",
    "豆腐",
    "茄子",
    "土豆",
    "西红柿",
    "鸡蛋",
    "莲藕",
    "香菇",
    "菜花",
    "虾仁",
    "鲈鱼",
    "白菜",
    "芹菜",
    "南瓜",
    "冬瓜",
    "山药",
]

METHODS = [
    "红烧",
    "清炒",
    "蒜香",
    "香煎",
    "干锅",
    "糖醋",
    "家常",
    "葱油",
    "椒盐",
    "酱香",
    "清蒸",
    "炖煮",
    "麻辣",
    "酸辣",
    "孜然",
    "砂锅",
    "凉拌",
]

CUISINES = ["川菜", "鲁菜", "粤菜", "苏菜", "浙菜", "闽菜", "湘菜", "徽菜", "家常菜"]

SEASONINGS = {
    "红烧": ["生抽", "老抽", "冰糖"],
    "清炒": ["食用油", "盐", "蒜"],
    "蒜香": ["蒜", "生抽", "盐"],
    "香煎": ["食用油", "黑胡椒", "盐"],
    "干锅": ["干辣椒", "豆瓣酱", "花椒"],
    "糖醋": ["香醋", "白糖", "生抽"],
    "清蒸": ["蒸鱼豉油", "葱", "姜"],
    "麻辣": ["辣椒", "花椒", "豆瓣酱"],
    "酸辣": ["香醋", "辣椒", "盐"],
    "孜然": ["孜然", "辣椒粉", "盐"],
}


def build_recipes() -> list[dict[str, object]]:
    recipes: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        for ingredient_index, ingredient in enumerate(INGREDIENTS):
            recipe_number = len(recipes) + 1
            seasonings = SEASONINGS.get(method, ["生抽", "盐", "葱姜"])
            recipes.append(
                {
                    "id": f"recipe-{recipe_number:03d}",
                    "title": f"{method}{ingredient}",
                    "cuisine": CUISINES[(method_index + ingredient_index) % len(CUISINES)],
                    "ingredients": [ingredient, *seasonings],
                    "steps": [
                        f"将{ingredient}清洗并完成切配备用。",
                        f"按{method}做法预处理锅具和调味料。",
                        f"加入{ingredient}与{'、'.join(seasonings)}，加热至入味。",
                        "根据口味调整盐度后装盘。",
                    ],
                    "description": f"一道以{ingredient}为主料的{method}家常菜。",
                    "tags": [method, ingredient, "中式食谱", "家常菜"],
                }
            )
    assert len(recipes) == 323
    return recipes


def build_queries(recipes: list[dict[str, object]]) -> list[dict[str, object]]:
    queries: list[dict[str, object]] = []
    for index in range(120):
        recipe = recipes[(index * 37) % len(recipes)]
        title = str(recipe["title"])
        ingredient = str(recipe["ingredients"][0])
        if index < 40:
            query = f"我想吃{ingredient}，推荐一道家常菜"
            query_type = "recommendation"
        elif index < 80:
            query = f"请问{title}需要什么食材"
            query_type = "ingredients"
        else:
            query = f"麻烦告诉我{title}怎么做一下"
            query_type = "steps"
        queries.append(
            {
                "id": f"query-{index + 1:03d}",
                "query": query,
                "query_type": query_type,
                "expected_recipe_ids": [recipe["id"]],
            }
        )
    return queries


def _ranking(recipe_ids: list[str], expected: str, *, hit: bool) -> list[str]:
    decoys = [recipe_id for recipe_id in recipe_ids if recipe_id != expected][:5]
    return [expected, *decoys[:4]] if hit else decoys


def build_snapshot(
    recipes: list[dict[str, object]], queries: list[dict[str, object]]
) -> dict[str, dict[str, list[str]]]:
    recipe_ids = [str(recipe["id"]) for recipe in recipes]
    baseline: dict[str, list[str]] = {}
    hybrid: dict[str, list[str]] = {}
    for index, query in enumerate(queries):
        query_id = str(query["id"])
        expected = str(query["expected_recipe_ids"][0])
        baseline[query_id] = _ranking(recipe_ids, expected, hit=index < 82)
        hybrid[query_id] = _ranking(recipe_ids, expected, hit=index < 106)
    return {"baseline": baseline, "hybrid": hybrid}


def build_demo_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    recipes = build_recipes()
    queries = build_queries(recipes)
    snapshot = build_snapshot(recipes, queries)
    outputs = {
        "recipes.json": recipes,
        "eval_queries.json": queries,
        "ranking_snapshot.json": snapshot,
    }
    for filename, value in outputs.items():
        (output_dir / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "src" / "recipe_rag" / "data"
    build_demo_data(output_dir)
    print(f"Generated 323 recipes and 120 evaluation queries in {output_dir}")


if __name__ == "__main__":
    main()
