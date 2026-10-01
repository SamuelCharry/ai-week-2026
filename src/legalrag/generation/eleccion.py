"""Elección de la letra en cerradas: permutaciones de las opciones y descarte en dos pasos.

Permutaciones (Zheng et al., ICLR 2024, "LLMs Are Not Robust Multiple Choice Selectors"): los modelos
prefieren ciertas letras por el token, no por el contenido. Se rota el orden de las opciones, se
calcula la probabilidad de cada letra en cada rotación y se promedia por *contenido*: la opción
"veinte días" recibe su promedio esté en la A o en la D. Todo es determinista (temperatura 0).

Descarte (Ma y Du, EMNLP 2023, "POE: Process of Elimination"): primero se puntúan todas las opciones,
se descartan las de menor promedio y se vuelve a preguntar solo con las finalistas. El descarte se
hace por puntaje y no pidiéndole al modelo que diga cuál está mal: esa forma rinde peor, sobre todo
en modelos pequeños (Balepur et al., ACL Findings 2024).
"""


def rotaciones(n):
    """Órdenes de las opciones: la identidad y sus n - 1 rotaciones."""
    return [list(range(i, n)) + list(range(i)) for i in range(n)]


def promedio_por_contenido(puntuar, opciones, permutar=True):
    """{letra original: probabilidad promedio} y el detalle de cada rotación.

    puntuar(opciones) -> {letra: probabilidad} para las opciones tal como se muestran al modelo.
    """
    letras = list(opciones)
    contenidos = [opciones[letra] for letra in letras]
    ordenes = rotaciones(len(letras)) if permutar else [list(range(len(letras)))]
    suma, detalle = dict.fromkeys(letras, 0.0), []
    for orden in ordenes:
        mostradas = {letras[posicion]: contenidos[indice] for posicion, indice in enumerate(orden)}
        probabilidades = puntuar(mostradas)
        for posicion, indice in enumerate(orden):
            suma[letras[indice]] += probabilidades[letras[posicion]]
        detalle.append({"orden": [letras[i] for i in orden], "probabilidades": probabilidades})
    return {letra: valor / len(ordenes) for letra, valor in suma.items()}, detalle


def elegir(puntuar, opciones, permutar=True, mantener=2):
    """(letra original elegida, registro). mantener=0 (o >= opciones) omite el descarte."""
    promedio, detalle = promedio_por_contenido(puntuar, opciones, permutar)
    registro = {"promedio": promedio, "rotaciones": detalle}
    if not mantener or not 1 < mantener < len(opciones):
        return max(sorted(promedio), key=promedio.get), registro
    finalistas = sorted(sorted(opciones, key=lambda letra: (-promedio[letra], letra))[:mantener])
    etiquetas = list(opciones)[:mantener]  # las finalistas se muestran como A, B...
    segunda, detalle2 = promedio_por_contenido(
        puntuar, {etiquetas[i]: opciones[f] for i, f in enumerate(finalistas)}, permutar)
    final = {f: segunda[etiquetas[i]] for i, f in enumerate(finalistas)}
    registro.update(descartadas=[l for l in opciones if l not in finalistas], finalistas=finalistas, final=final,
                    rotaciones_finalistas=detalle2)
    return max(sorted(final), key=final.get), registro
