class RAGServiceError(Exception):
    """Base exception for application-level RAG service failures."""


class RetrievalError(RAGServiceError):
    """Raised when retrieval fails because the vector store is unavailable or misconfigured."""


class GenerationError(RAGServiceError):
    """Raised when answer generation fails because the LLM provider fails."""


class EmptyRetrievalError(RAGServiceError):
    """Raised when retrieval succeeds but returns no usable chunks."""


class InvalidFiltersError(RAGServiceError):
    """Raised when request filters are invalid or unsupported."""
