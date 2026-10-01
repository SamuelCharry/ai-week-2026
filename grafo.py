from legalrag.config import CONFIG
from legalrag.ingestion.gap_analysis import audit


if __name__ == "__main__":
    audit(CONFIG)
