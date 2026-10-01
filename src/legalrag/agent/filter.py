from legalrag.citations.extract import fold


def scope(question):
    area = fold(question.get("area") or "")
    outside = area in {"derecho ambiental", "derecho internacional", "derecho internacional publico",
                       "derecho internacional privado"}
    text = fold(question.get("pregunta", ""))
    flags = [term for term in ("derecho ambiental", "derecho internacional", "tratado internacional",
                               "licencia ambiental") if term in text]
    # Una mención ambiental puede formar parte de una pregunta constitucional válida.
    return {"fuera_de_alcance": outside, "revisar_alcance": bool(flags) and not outside,
            "senales": flags}
