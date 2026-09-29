"""Áreas del banco y alcance de las normas que entran sin clasificar (ronda 5, árbol del Senado).

El índice del Senado solo da "LEY 405 de 1997": el área sale del epígrafe ("Por la cual se
...") que abre el texto descargado. Las reglas son palabras clave por área, en orden; una
norma puede caer en varias áreas y, si ninguna aplica, queda en derecho administrativo (la
mayoría de leyes sin materia privada organizan el Estado).

Además marca el alcance, sin sacar nada del corpus, para que la ingesta o el índice puedan
filtrar:
    tratado_internacional  ley aprobatoria de tratado (el banco excluye derecho internacional)
    honorifica             ley de honores, conmemoraciones o estampillas
    ambiental              materia ambiental (el banco la excluye)

Solo toca documentos sin áreas o de origen senado_arbol; reescribe data/raw/manifest.json.
Correr con la descarga detenida (scripts.corpus.reconstruir escribe el mismo manifiesto).

Uso: python -m scripts.corpus.areas [--dry-run]
"""
import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.corpus.grafo import texto_en_cache  # noqa: E402

# (slug del área, patrón sobre el epígrafe en minúsculas y sin tildes)
REGLAS = [
    ("tributario", r"impuest|tribut|\brenta\b|\biva\b|gravamen|contribucion|\bdian\b|aduan|retencion en la fuente|timbre|"
                   r"predial|industria y comercio|sobretasa|estampilla"),
    ("laboral", r"trabaj|laboral|empleo|salari|pension|seguridad social|cesantia|prestaciones sociales|sindic|"
                r"riesgos (laborales|profesionales)|subsidio familiar|carrera administrativa"),
    ("penal", r"penal|delito|\bpenas?\b|carcel|penitenci|crimin|estupefacientes|extincion de dominio|narcotrafic|"
              r"terroris|lavado de activos|corrupcion"),
    ("procesal", r"procedimiento|\bproceso|procesal|jurisdicci|juzgad|jueces|conciliaci|arbitraje|administracion de justicia|"
                 r"accion de tutela|acciones populares|rama judicial|casacion|descongestion"),
    ("comercial", r"sociedad|comerci|mercantil|empresa|insolvencia|reorganizacion|financier|bancari|seguros|mercado de valores|"
                  r"camaras? de comercio|contador|titulos valores|cooperativ|emprendimiento|factura"),
    ("civil", r"civil|contrato|propiedad horizontal|bienes|registro|notari|arrendamiento|vivienda|obligaciones|sucesion|"
              r"responsabilidad civil|tierras|baldios|catastro|propiedad raiz"),
    ("familia", r"familia|matrimonio|menores? de edad|nin[oa]s|infancia|adolescen|adopci|alimentos|mujer|violencia intrafamiliar|"
                r"divorcio|union marital|adulto mayor|discapacidad|genero|paternidad"),
    ("mercados", r"consumidor|competencia desleal|libre competencia|practicas comerciales restrictivas|datos personales|"
                 r"habeas data|propiedad intelectual|derechos? de autor|propiedad industrial|marcas?\b|patente|publicidad|"
                 r"telecomunicaciones|tecnologias de la informacion|comercio electronico|proteccion de datos"),
    ("constitucional", r"estatutaria|reforma constitucional|constitucion politica|derechos fundamentales|participacion ciudadana|"
                       r"referendo|plebiscito|partidos politicos|mecanismos de participacion|acto legislativo|paz|victimas"),
    ("administrativo", r"administrativ|contratacion (estatal|publica)|funcion publica|servidores publicos|municipi|departament|"
                       r"ministerio|presupuest|regalias|contralor|disciplinari|policia|servicios publicos|entidades? territorial|"
                       r"ordenamiento territorial|electoral|salud|educacion|transporte|planeacion|descentralizacion"),
]
REGLAS = [(slug, re.compile(p)) for slug, p in REGLAS]

# Solo la ley que aprueba el instrumento; la que lo desarrolla o reglamenta es derecho interno.
TRATADO = re.compile(r"^(por (medio de )?(la|el) cual )?se aprueb(a|an)\b.{0,160}\b(convenio|convencion|tratado|acuerdo|"
                     r"protocolo|pacto|enmienda|estatuto|memorando|carta|constitucion)\b")
HONORIFICA = re.compile(r"se asocia a la (celebracion|conmemoracion)|rinde (publico )?homenaje|honores|conmemora|exalta|"
                        r"declara .{0,80}patrimonio (cultural|historico)|aniversario|se reconoce .{0,60}(festival|fiesta|reinado)")
AMBIENTAL = re.compile(r"ambient|paramo|parques nacionales|biodiversidad|recursos naturales renovables|cambio climatico|"
                       r"fauna|flora|forestal|humedal|residuos|emisiones")


def plano(texto):
    texto = unicodedata.normalize("NFD", texto.lower())
    return re.sub(r"\s+", " ", "".join(c for c in texto if unicodedata.category(c) != "Mn"))


def epigrafe(texto):
    """"Por la cual se ..." hasta "el congreso ... decreta" o el primer artículo (máx. 1.500).

    Senado intercala anotaciones entre <> ("<ley inexequible, sentencia c-...>") y notas de
    vigencia antes del epígrafe; se quitan para que no aporten áreas ajenas a la norma.
    """
    cabeza = re.sub(r"<[^>]*>", " ", plano(texto[:8000]))
    inicio = re.search(r"\bpor (medio de )?(la|el) cual\b|\bpor la que\b|\bsobre\b|\bmediante (la|el) cual\b", cabeza)
    cabeza = cabeza[inicio.start():] if inicio else cabeza
    corte = re.search(r"\bel congreso de (colombia|la republica)\b|\bel presidente de la republica\b|\bdecreta\s*:|"
                      r"\bresumen de notas\b|\bjurisprudencia vigencia\b|\barticulo (1|primero|unico)\b", cabeza)
    return cabeza[:corte.start()][:1500] if corte else cabeza[:1500]


def clasificar(texto):
    ep = epigrafe(texto)
    areas = [slug for slug, patron in REGLAS if patron.search(ep)] or ["administrativo"]
    alcance = "tratado_internacional" if TRATADO.search(ep) else "honorifica" if HONORIFICA.search(ep) else \
        "ambiental" if AMBIENTAL.search(ep) and not set(areas) - {"administrativo"} else None
    return areas, alcance


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="solo muestra el resumen, no escribe")
    args = ap.parse_args()
    ruta = RAIZ / "data/raw/manifest.json"
    manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
    conteo, alcances = collections.Counter(), collections.Counter()
    for entrada in manifiesto:
        if entrada.get("areas") and entrada.get("origen_ampliacion") != "senado_arbol":
            continue
        if entrada.get("tipo") not in ("ley", "decreto", "acto_legislativo"):
            continue
        areas, alcance = clasificar(texto_en_cache(RAIZ, entrada))
        # Las áreas ya declaradas (inventario curado) se conservan; se suman las del epígrafe.
        entrada["areas"] = sorted(set(entrada.get("areas") or []) | set(areas))
        entrada["areas_por_epigrafe"] = True
        if alcance:
            entrada["alcance"] = alcance
        conteo.update(areas)
        alcances[alcance or "general"] += 1
    print("áreas:", dict(conteo.most_common()))
    print("alcance:", dict(alcances.most_common()))
    if not args.dry_run:
        ruta.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{ruta.relative_to(RAIZ)} actualizado")


if __name__ == "__main__":
    main()
