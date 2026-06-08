from typing import Literal

RetrievalMode = Literal[
    "dense",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
]

EvalRetrievalMode = Literal[
    "dense",
    "lexical",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
]

DEFAULT_RETRIEVAL_MODE: RetrievalMode = "dense"

SUPPORTED_RETRIEVAL_MODES: set[str] = {
    "dense",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
}

SUPPORTED_EVAL_RETRIEVAL_MODES: set[str] = {
    "dense",
    "lexical",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
}
