"""Ventanas de búsqueda por unidad: tokens del encoder, solapamiento y cabecera literal."""
from legalrag.comun import hash_json

VERSION_TEXTO_BUSQUEDA = 2


def metadatos_busqueda(unidad):
    articulo = "" if unidad.get("articulo") is None else "Artículo " + str(unidad["articulo"])
    seccion = unidad.get("seccion")
    encabezado = f"{seccion}\n" if seccion else ""
    return f"{unidad['titulo']}\n{encabezado}{articulo}\n"


def crear_ventanas(unidades, encoder, tamano=320, solapamiento=48):
    if not 0 <= solapamiento < tamano:
        raise ValueError("Solapamiento fuera de rango")
    ventanas = []
    for unidad in unidades:
        texto = unidad["texto"]
        cabecera = metadatos_busqueda(unidad)
        margen = len(encoder.tokenizer(encoder.prefijos["passage"] + cabecera)["input_ids"]) + 4
        limite = min(tamano, encoder.limite - margen)
        if limite <= solapamiento:
            raise ValueError("Metadatos demasiado largos para el encoder")
        offsets = encoder.tokenizer(texto, add_special_tokens=False, truncation=False,
                                    return_offsets_mapping=True)["offset_mapping"]
        offsets = [(a, b) for a, b in offsets if b > a]
        if not offsets:
            continue
        posicion = 0
        while posicion < len(offsets):
            fin_token = min(posicion + limite, len(offsets))
            inicio = 0 if posicion == 0 else offsets[posicion][0]
            fin = len(texto) if fin_token == len(offsets) else offsets[fin_token][0]
            if fin <= inicio:
                raise ValueError("Ventana vacía")
            pasaje = texto[inicio:fin]
            while len(encoder.tokenizer(encoder.prefijos["passage"] + cabecera + pasaje)["input_ids"]) > encoder.limite:
                fin_token -= 1
                fin = offsets[fin_token][0]
                pasaje = texto[inicio:fin]
            fila = dict(unidad)
            fila.update(inicio=unidad["inicio"] + inicio, fin=unidad["inicio"] + fin,
                        unidad_inicio=unidad["inicio"], unidad_fin=unidad["fin"], texto=pasaje,
                        texto_busqueda=cabecera + pasaje, recuperar_unidad_completa=True)
            fila["fragmento_id"] = hash_json([unidad["unidad_id"], fila["inicio"], fila["fin"]])[:24]
            ventanas.append(fila)
            if fin_token == len(offsets):
                break
            posicion = max(posicion + 1, fin_token - solapamiento)
    return ventanas
