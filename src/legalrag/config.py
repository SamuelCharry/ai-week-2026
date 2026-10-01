from dataclasses import asdict, dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "Qwen/Qwen3-8B"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    llm_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    device: str = "cuda"
    batch_size: int = 8
    reranker_batch_size: int = 6
    window_tokens: int = 512
    overlap_tokens: int = 64
    embedding_max_tokens: int = 1024
    top_k_retrieval: int = 50
    top_k_evidence: int = 10
    context_tokens: int = 5500
    max_input_tokens: int = 6500
    max_new_tokens: int = 700
    min_reranker_score: float = 0.25
    rrf_k: int = 60
    seed: int = 17
    hybrid: bool = False
    use_reranker: bool = True
    use_hyde: bool = True
    mc_thinking: bool = True
    mc_thinking_max_tokens: int = 768
    augment_max: int = 6
    augment_min_rerank: float = 0.15
    norm_hint: bool = False
    multi_query: bool = False
    multi_query_threshold: float = 0.4
    # Cerberus Mark 43: herramientas corregidas (SMLMV 2026, UVT, años, liquidación) y normalizador de citas.
    herramientas_v2: bool = False
    normalizador_citas: bool = False
    max_params: int = 9_000_000_000

    @property
    def prepared(self):
        return self.root / "data/processed/corpus_preparado"

    @property
    def data_dir(self):
        return self.root / "data"

    @property
    def index_dir(self):
        return self.root / "index"

    @property
    def reports(self):
        return self.root / "reports"

    @property
    def questions_file(self):
        return self.data_dir / "oficial/data/sample_50.jsonl"

    @property
    def schema_file(self):
        return self.data_dir / "oficial/schema/submission.schema.json"

    @property
    def output_file(self):
        return self.root / "submissions.jsonl"

    def serializable(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

    @classmethod
    def from_env(cls):
        return cls(root=Path(os.environ.get("LEGALRAG_ROOT", ROOT)).resolve())


CONFIG = Config.from_env()
DATA_DIR = CONFIG.data_dir
INDEX_DIR = CONFIG.index_dir
OUTPUT_FILE = CONFIG.output_file
QUESTIONS_FILE = CONFIG.questions_file
EMBEDDING_MODEL = CONFIG.embedding_model
LLM_MODEL = CONFIG.llm_model
RERANKER_MODEL = CONFIG.reranker_model
TOP_K_RETRIEVAL = CONFIG.top_k_retrieval
TOP_K_RERANK = CONFIG.top_k_evidence
BATCH_SIZE = CONFIG.batch_size
DEVICE = CONFIG.device
