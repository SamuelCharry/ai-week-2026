import argparse
from dataclasses import replace
from pathlib import Path
from legalrag.config import CONFIG


def main():
    from legalrag.silencio import silenciar
    silenciar()
    parser = argparse.ArgumentParser(description="Preguntas jurídicas con corpus colombiano")
    parser.add_argument("--root", type=Path, default=CONFIG.root)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--questions", type=Path)
    audit.add_argument("--coverage", action="store_true")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--replace", action="store_true")
    sub.add_parser("package-raw")
    sub.add_parser("profile")
    sub.add_parser("finalize")
    index = sub.add_parser("index")
    index.add_argument("--replace", action="store_true")
    index.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    run = sub.add_parser("run")
    run.add_argument("--questions", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--expected-count", type=int)
    retrieval = run.add_mutually_exclusive_group()
    retrieval.add_argument("--dense-only", action="store_true")
    retrieval.add_argument("--hybrid", action="store_true")
    run.add_argument("--no-reranker", action="store_true")
    run.add_argument("--no-hyde", action="store_true")
    run.add_argument("--mc-thinking", action="store_true",
                     help="activa el modo 'thinking' de Qwen3 solo para preguntas cerradas")
    run.add_argument("--multi-query", action="store_true",
                     help="descompone la pregunta en sub-consultas cuando el retrieval es debil")
    run.add_argument("--herramientas-v2", action="store_true",
                     help="Mark 43: SMLMV 2026 correcto, UVT, dos años si la pregunta no trae año y plazos de "
                          "liquidacion solo si se habla de liquidar un contrato")
    run.add_argument("--normalizador", action="store_true",
                     help="Mark 43: avisa si la pregunta u opcion cita una ley con el ano equivocado")
    run.add_argument("--augment-max", type=int, default=CONFIG.augment_max,
                     help="maximo de normas respaldadas que se agregan a la respuesta")
    run.add_argument("--threshold", type=float, default=CONFIG.min_reranker_score)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--gold", type=Path)
    evaluation.add_argument("--official", action="store_true")
    evaluation.add_argument("--ragas", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = replace(CONFIG, root=args.root.resolve())
    if args.command == "profile":
        from legalrag.ingestion.profile import profile
        profile(config)
    elif args.command == "prepare":
        from legalrag.ingestion.prepare_raw import prepare_raw
        prepare_raw(config, replace=args.replace)
    elif args.command == "package-raw":
        from legalrag.ingestion.package_raw import package_raw
        package_raw(config)
    elif args.command == "audit":
        from legalrag.ingestion.gap_analysis import audit
        audit(config)
        if args.coverage:
            from legalrag.ingestion.coverage_audit import coverage_audit
            coverage_audit(config, args.questions or config.questions_file)
    elif args.command == "finalize":
        from legalrag.ingestion.finalize import finalize
        finalize(config)
    elif args.command == "index":
        if args.batch_size < 1:
            parser.error("--batch-size debe ser positivo")
        from legalrag.indexing.build_index import build_index
        build_index(replace(config, batch_size=args.batch_size), replace=args.replace)
    elif args.command == "run":
        if not 0 <= args.threshold <= 1:
            parser.error("--threshold debe estar entre 0 y 1")
        from legalrag.pipeline import run
        config = replace(config, hybrid=args.hybrid, use_reranker=not args.no_reranker,
                         use_hyde=not args.no_hyde, min_reranker_score=args.threshold,
                         mc_thinking=args.mc_thinking, augment_max=args.augment_max,
                         multi_query=args.multi_query, herramientas_v2=args.herramientas_v2,
                         normalizador_citas=args.normalizador)
        run(config, args.questions, args.output or config.output_file, args.resume, args.expected_count)
    elif args.command == "evaluate":
        from legalrag.evaluation.evaluate import evaluate
        evaluate(config, args.predictions, args.gold, args.official, args.ragas)
    elif args.command == "serve":
        from legalrag.server import serve
        serve(config, args.port)


if __name__ == "__main__":
    main()
