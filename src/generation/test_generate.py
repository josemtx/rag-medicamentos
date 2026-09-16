from generate import build_prompt


def test_build_prompt_includes_source_labels_and_query():
    chunks = [{"medicamento": "IBUPROFENO X", "titulo": "Posologia", "doc_type": "ficha_tecnica", "text": "400mg cada 8h"}]
    prompt = build_prompt("dosis de ibuprofeno", chunks)
    assert "IBUPROFENO X" in prompt
    assert "Posologia" in prompt
    assert "400mg cada 8h" in prompt
    assert "dosis de ibuprofeno" in prompt


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_"):
            _fn()
    print("OK")
