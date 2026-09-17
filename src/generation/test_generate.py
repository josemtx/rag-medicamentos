from generate import build_prompt


def test_build_prompt_includes_source_labels_and_query():
    chunks = [{"medicamento": "IBUPROFENO X", "titulo": "Posologia", "doc_type": "ficha_tecnica",
               "text": "400mg cada 8h", "principios_activos": "IBUPROFENO 400 mg"}]
    prompt = build_prompt("dosis de ibuprofeno", chunks)
    assert "IBUPROFENO X" in prompt
    assert "Posologia" in prompt
    assert "400mg cada 8h" in prompt
    assert "dosis de ibuprofeno" in prompt


def test_build_prompt_exposes_composition():
    """Sin la composición, "no tomar con medicamentos que contengan X" se malinterpreta
    cuando el propio medicamento ya lleva X."""
    chunks = [{"medicamento": "ANTIDOL DUAL", "titulo": "Interacciones", "doc_type": "ficha_tecnica",
               "text": "No tomar con otros medicamentos que contengan ibuprofeno.",
               "principios_activos": "PARACETAMOL 500 mg, IBUPROFENO 200 mg"}]
    prompt = build_prompt("puedo combinarlos?", chunks)
    assert "PARACETAMOL 500 mg, IBUPROFENO 200 mg" in prompt


def test_build_prompt_handles_missing_composition():
    chunks = [{"medicamento": "X", "titulo": "T", "doc_type": "prospecto", "text": "t"}]
    assert "no consta" in build_prompt("q", chunks)


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_"):
            _fn()
    print("OK")
