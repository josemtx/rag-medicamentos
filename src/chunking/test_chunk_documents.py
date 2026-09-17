from chunk_documents import html_to_text, split_long_text, chunk_document


def test_html_to_text_strips_tags_and_collapses_whitespace():
    html = "<div>\r\n  <p style=\"x\"><span>Hola   mundo.</span></p>\r\n</div>"
    assert html_to_text(html) == "Hola mundo."


def test_split_long_text_respects_max_chars():
    text = "Frase uno. Frase dos. Frase tres. Frase cuatro."
    chunks = split_long_text(text, max_chars=20)
    assert all(len(c) <= 20 or " " not in c for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_chunk_document_splits_oversized_section():
    sections = [{"seccion": "1", "titulo": "T", "contenido": "<p>" + ("Palabra. " * 500) + "</p>"}]
    chunks = chunk_document("123", "ficha_tecnica", sections, "MED X", "IBUPROFENO 400 mg")
    assert len(chunks) > 1
    assert all(c["nregistro"] == "123" and c["doc_type"] == "ficha_tecnica" for c in chunks)


def test_chunk_document_skips_empty_section():
    sections = [{"seccion": "1", "titulo": "T", "contenido": "<p></p>"}]
    assert chunk_document("123", "prospecto", sections, "MED X", "IBUPROFENO 400 mg") == []


def test_chunk_document_skips_placeholder_section():
    sections = [{"seccion": "4", "titulo": "DATOS CLINICOS", "contenido": "<p>.</p>"}]
    assert chunk_document("123", "ficha_tecnica", sections, "MED X", "IBUPROFENO 400 mg") == []


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_"):
            _fn()
    print("OK")
