from rerank import rerank


class _FakeCrossEncoder:
    """Puntúa más alto los textos que contienen 'ibuprofeno'."""

    def predict(self, pairs):
        return [1.0 if "ibuprofeno" in text.lower() else 0.0 for _, text in pairs]


def test_rerank_sorts_by_cross_encoder_score_desc():
    candidates = [
        {"text": "Paracetamol 500mg", "score": 0.9},
        {"text": "Ibuprofeno 600mg dosis", "score": 0.1},
        {"text": "Naproxeno sodico", "score": 0.5},
    ]
    result = rerank("query", candidates, _FakeCrossEncoder(), k=2)
    assert len(result) == 2
    assert result[0]["text"] == "Ibuprofeno 600mg dosis"
    assert result[0]["score_rerank"] == 1.0


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_"):
            _fn()
    print("OK")
