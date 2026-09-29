"""Renderiza el reporte Markdown y una figura del censo; no procesa modelos."""
from pathlib import Path
import html
import json
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[3]


def main():
    folder = ROOT / 'reports/reporte_evaluacion'
    folder.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((ROOT / 'reports/perfil_corpus_preparado/resumen_tamanos.json').read_bytes())
    rows = metrics['por_tipo']
    shown = [r for r in rows if r['tipo'] in {'sentencia', 'ley', 'decreto', 'compendio', 'concepto', 'circular'}]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout='constrained')
    fig.patch.set_facecolor('#faf9f6')
    for ax in axes:
        ax.set_facecolor('#faf9f6')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#b7c1c7')
        ax.tick_params(colors='#34464c', labelsize=10)
        ax.grid(axis='x', color='#d8dfdf', linewidth=.7, alpha=.7)
        ax.set_axisbelow(True)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda n, _: f'{n:,.0f}'.replace(',', '.')))
    labels = [r['tipo'].capitalize() for r in shown]
    positions = list(range(len(shown)))
    axes[0].barh(positions, [r['documentos'] for r in shown], color='#236a79', height=.6)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(1, 30000)
    axes[0].set_title('Composición por tipo', loc='left', fontsize=14, fontweight='bold', pad=18)
    axes[0].set_xlabel('Documentos · escala logarítmica')
    for y, row in enumerate(shown):
        axes[0].text(row['documentos'] * 1.13, y, f"{row['documentos']:,}".replace(',', '.'), va='center', fontsize=10)
    for y, row in enumerate(shown):
        axes[1].plot([row['p50_palabras'], row['p95_palabras']], [y, y], color='#b0bbbd', linewidth=3)
    axes[1].scatter([r['p50_palabras'] for r in shown], positions, color='#236a79', s=50, label='Mediana', zorder=3)
    axes[1].scatter([r['p95_palabras'] for r in shown], positions, color='#b76d2a', s=50, label='Percentil 95', zorder=3)
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(400, 220000)
    axes[1].set_title('Longitud: la cola importa', loc='left', fontsize=14, fontweight='bold', pad=18)
    axes[1].set_xlabel('Palabras regex por documento · escala logarítmica')
    axes[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.19), ncol=2)
    fig.suptitle('Censo de 13.967 documentos · seis tipos seleccionados', x=.02, ha='left', fontsize=17, fontweight='bold', color='#12313b')
    fig.savefig(folder / 'perfil_resumen.svg', bbox_inches='tight')
    fig.savefig(folder / 'perfil_resumen.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    source = ROOT / 'docs/REPORTE_CORPUS_Y_EXPERIMENTOS.md'
    md = MarkdownIt('commonmark', {'html': False}).enable('table')
    tokens = md.parse(source.read_text(encoding='utf-8'))
    nav, seen = [], set()
    for i, token in enumerate(tokens):
        if token.type != 'heading_open':
            continue
        title = tokens[i + 1].content
        stem = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode().lower()
        stem = re.sub(r'[^a-z0-9]+', '-', stem).strip('-')
        anchor, suffix = stem, 2
        while anchor in seen:
            anchor = f'{stem}-{suffix}'
            suffix += 1
        seen.add(anchor)
        token.attrSet('id', anchor)
        if token.tag == 'h2':
            nav.append(f'<a href="#{anchor}">{html.escape(title)}</a>')
    rendered = md.renderer.render(tokens, md.options, {})
    css = '''
    :root{color-scheme:light;--ink:#173039;--muted:#52636a;--accent:#176476;--line:#d5dfe1}
    *{box-sizing:border-box}body{margin:0;background:#f6f7f6;color:var(--ink);font:16px/1.7 system-ui,Segoe UI,sans-serif}
    header{padding:20px 5vw;background:#12313b;color:#fff;font-size:13px;letter-spacing:.08em}
    .layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:38px;max-width:1500px;margin:auto;padding:38px}
    nav{position:sticky;top:25px;align-self:start;font-size:13px;line-height:1.45}nav a{display:block;padding:8px 0;text-decoration:none;border-bottom:1px solid var(--line)}
    main{min-width:0;background:white;padding:35px 45px;border:1px solid var(--line);border-radius:8px}
    h1{font-size:36px;line-height:1.2;margin-top:0}h2{font-size:25px;line-height:1.3;margin-top:2.2em;padding-top:12px;border-top:2px solid var(--accent)}h3{font-size:20px;margin-top:1.8em}
    a{color:var(--accent);text-underline-offset:3px;overflow-wrap:anywhere}p,li{max-width:100ch}li{margin:.4em 0}strong{font-weight:650}
    table{display:block;overflow-x:auto;border-collapse:collapse;font-size:13px;line-height:1.55;margin:24px 0;width:100%}thead{background:#eaf1f2}th,td{text-align:left;padding:11px 13px;border:1px solid var(--line);vertical-align:top;min-width:105px}tr:nth-child(even){background:#f8fafb}
    pre{overflow:auto;padding:18px;background:#102e37;color:#e9f3f4;border-radius:5px;font-size:13px}code{font-size:.88em;overflow-wrap:anywhere}p code,li code{background:#edf2f3;padding:2px 4px}
    img{max-width:100%;height:auto}blockquote{border-left:3px solid var(--accent);padding-left:20px;color:var(--muted)}
    @media(max-width:950px){.layout{display:block;padding:15px}nav{position:static;margin-bottom:25px;columns:2}main{padding:24px}h1{font-size:29px}}
    @media print{body{background:#fff;font-size:10pt}header,nav{display:none}.layout{display:block;padding:0}main{border:none;padding:0}h2{break-after:avoid}table{display:table;font-size:8pt}tr{break-inside:avoid}a{color:inherit}pre{white-space:pre-wrap}h1{font-size:24pt}}
    '''
    target = ROOT / 'docs/REPORTE_CORPUS_Y_EXPERIMENTOS.html'
    target.write_text('<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Corpus y experimentos · AI Week 2026</title><style>' + css + '</style></head><body><header>AI WEEK 2026 · REPORTE INTERNO · 29 SEPTIEMBRE 2026</header><div class="layout"><nav aria-label="Contenido">' + ''.join(nav) + '</nav><main>' + rendered + '</main></div></body></html>', encoding='utf-8')
    print(json.dumps({'html': str(target), 'secciones': len(nav), 'figura': str(folder / 'perfil_resumen.svg')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
