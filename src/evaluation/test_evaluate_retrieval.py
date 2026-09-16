from evaluate_retrieval import gold_rank, metrics


def test_gold_rank_finds_position():
    results = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    assert gold_rank(results, "a") == 1
    assert gold_rank(results, "c") == 3


def test_gold_rank_returns_none_when_missing():
    assert gold_rank([{"chunk_id": "a"}], "z") is None


def test_metrics_perfect_retrieval():
    m = metrics([1, 1, 1, 1])
    assert m["recall@1"] == 1.0 and m["mrr"] == 1.0


def test_metrics_counts_thresholds_and_misses():
    # rangos: 1 (dentro de todos), 5 (fuera de R@1), 10 (solo en R@10), None (fallo)
    m = metrics([1, 5, 10, None])
    assert m["recall@1"] == 0.25
    assert m["recall@5"] == 0.5
    assert m["recall@10"] == 0.75
    assert abs(m["mrr"] - (1 + 1 / 5 + 1 / 10) / 4) < 1e-9


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_"):
            _fn()
    print("OK")
