import numpy as np
from hybrid_search import _normalize, _rrf_scores, tokenize


def test_rrf_assigns_best_score_to_highest_ranked():
    scores = np.array([0.1, 0.9, 0.5])
    rrf = _rrf_scores(scores, k_rrf=60)
    # el mejor score (indice 1) debe recibir posicion 1 -> 1/61
    assert abs(rrf[1] - 1 / 61) < 1e-12
    assert abs(rrf[2] - 1 / 62) < 1e-12
    assert abs(rrf[0] - 1 / 63) < 1e-12


def test_rrf_ignores_score_magnitude():
    # misma ordenacion, escalas radicalmente distintas -> mismo resultado
    a = _rrf_scores(np.array([1.0, 2.0, 3.0]), k_rrf=60)
    b = _rrf_scores(np.array([0.001, 500.0, 900.0]), k_rrf=60)
    assert np.allclose(a, b)


def test_normalize_scales_to_0_1():
    scores = np.array([1.0, 3.0, 5.0])
    result = _normalize(scores)
    assert result[0] == 0.0 and result[-1] == 1.0 and result[1] == 0.5


def test_normalize_constant_scores_returns_zeros():
    scores = np.array([2.0, 2.0, 2.0])
    assert (_normalize(scores) == 0).all()


def test_tokenize_lowercases_and_splits():
    assert tokenize("Ibuprofeno 600mg, dosis.") == ["ibuprofeno", "600mg", "dosis"]


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_"):
            _fn()
    print("OK")
