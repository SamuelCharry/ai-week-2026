import argparse
from pathlib import Path
from legalrag.config import CONFIG
from legalrag.ingestion.coverage_audit import coverage_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=CONFIG.questions_file)
    args = parser.parse_args()
    coverage_audit(CONFIG, args.questions)
