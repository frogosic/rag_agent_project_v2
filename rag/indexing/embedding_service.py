import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Creates dense vector embeddings for text chunks."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        logger.info("Loading embedding model: %s", model_name)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        embedding_dimension = self.model.get_embedding_dimension()
        if embedding_dimension is None:
            raise ValueError(
                f"Model {model_name!r} did not report an embedding dimension."
            )

        self._dimension = embedding_dimension

    @property
    def dimension(self) -> int:
        """Return the dimension of embedding vectors produced by this model."""
        return self._dimension

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts into dense vectors."""
        if not texts:
            return []

        logger.info("Embedding %s texts with model: %s", len(texts), self.model_name)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed_texts([text])[0]
