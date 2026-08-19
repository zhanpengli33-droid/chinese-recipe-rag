import pytest

from recipe_rag.models import Recipe


@pytest.fixture
def sample_recipes() -> list[Recipe]:
    return [
        Recipe(
            id="recipe-a",
            title="西红柿炒鸡蛋",
            cuisine="家常菜",
            ingredients=("西红柿", "鸡蛋", "盐", "糖"),
            steps=("西红柿切块。", "鸡蛋炒熟。", "合炒调味。"),
            description="酸甜家常快手菜。",
            tags=("快手菜", "素菜"),
        ),
        Recipe(
            id="recipe-b",
            title="红烧牛腩",
            cuisine="川菜",
            ingredients=("牛腩", "土豆", "生抽", "冰糖"),
            steps=("牛腩焀水。", "炒糖色。", "小火炖煮。"),
            description="酱香浓郁的炖菜。",
            tags=("红烧", "肉菜"),
        ),
        Recipe(
            id="recipe-c",
            title="蓜蓉西兰花",
            cuisine="粤菜",
            ingredients=("西兰花", "蒜", "盐"),
            steps=("西兰花焂水。", "蒜末爆香。", "快速翻炒。"),
            description="清爽蓜香素菜。",
            tags=("清淡", "素菜"),
        ),
    ]
