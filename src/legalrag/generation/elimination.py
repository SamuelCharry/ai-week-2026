def instruction(question):
    if question.get("formato") != "multiple_choice":
        return ""
    return (
        "Evalúa cada opción frente al contexto. Descarta solo las opciones contradichas por la "
        "evidencia. Escribe una razón breve para cada descarte en descarte_opciones. "
        "Selecciona una opción únicamente si el contexto permite sostenerla. "
        "Si quedan alternativas sin resolver, devuelve abstencion=true. "
        "No añadas razonamientos internos ni elijas por intuición."
    )
