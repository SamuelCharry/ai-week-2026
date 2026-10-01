def expand(question, generator):
    messages = [
        {"role": "system", "content": (
            "Escribe un breve pasaje hipotético para buscar evidencia jurídica colombiana. "
            "Describe conceptos y requisitos pertinentes. No inventes números de leyes, "
            "artículos, sentencias ni hechos. No respondas las opciones. Máximo 120 palabras. "
            "Este texto sirve únicamente para buscar documentos.")},
        {"role": "user", "content": question},
    ]
    return generator.complete(messages, max_new_tokens=200)
