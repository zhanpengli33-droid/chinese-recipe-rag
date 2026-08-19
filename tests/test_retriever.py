from recipe_rag.retriever import HybridRetriever, reciprocal_rank_fusion, tokenize


def test_chinese_tokenizer_includes_unigrams_and_bigrams() -> None:
    tokens = tokenize("红烧牛腩 beef")
    assert "红" in tokens
    assert "红烧" in tokens
    assert "beef" in tokens


def test_rrf_rewards_items_seen_in_multiple_rankings() -> None:
    fused = reciprocal_rank_fusion((("a", "b", "c"), ("a", "c", "d")))
    assert fused[0][0] == "a"


def test_hybrid_search_finds_exact_recipe(sample_recipes) -> None:
    retriever = HybridRetriever(sample_recipes)
    hits = retriever.search("红烧牛腩怎么做", top_k=2)
    assert hits[0].recipe.id == "recipe-b"
    assert hits[0].lexical_score > 0


def test_retriever_rejects_invalid_inputs(sample_recipes) -> None:
    retriever = HybridRetriever(sample_recipes)
    for query, top_k in (("", 5), ("牛腩", 0)):
        try:
            retriever.search(query, top_k=top_k)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
