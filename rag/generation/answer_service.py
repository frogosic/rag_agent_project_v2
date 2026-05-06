import logging
from typing import Any

from anthropic.types import MessageParam

from rag.generation.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """
You are a grounded RAG answer generator.

Use only the provided context chunks to answer the user's question.
If the context does not contain enough information, say that the available context is insufficient.
Do not invent facts.
When possible, cite the source names used in the answer.
""".strip()


class AnswerService:
    """Generate grounded answers from retrieved chunks."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.max_tokens = max_tokens

    def answer(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        """Generate an answer using the query and retrieved chunks."""
        if not retrieved_chunks:
            return "I do not have enough retrieved context to answer this question."

        logger.info(
            "Generating answer for query using %s retrieved chunks",
            len(retrieved_chunks),
        )

        context = self._format_context(retrieved_chunks)

        messages: list[MessageParam] = [
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Context chunks:\n{context}\n\n"
                    "Answer the question using only the context chunks above."
                ),
            }
        ]

        return self.llm_client.complete(
            messages=messages,
            max_tokens=self.max_tokens,
            system=DEFAULT_SYSTEM_PROMPT,
        )

    @staticmethod
    def _format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks for the LLM prompt."""
        formatted_chunks: list[str] = []

        for index, result in enumerate(retrieved_chunks, start=1):
            payload = result.get("payload") or {}

            source = payload.get("source")
            content_type = payload.get("content_type")
            domain = payload.get("domain")
            doc_role = payload.get("doc_role")
            chunk_id = payload.get("chunk_id")
            text = payload.get("text", "")
            score = result.get("score")

            formatted_chunks.append(
                "\n".join(
                    [
                        f"[Chunk {index}]",
                        f"source: {source}",
                        f"chunk_id: {chunk_id}",
                        f"score: {score}",
                        f"content_type: {content_type}",
                        f"domain: {domain}",
                        f"doc_role: {doc_role}",
                        "text:",
                        text,
                    ]
                )
            )

        return "\n\n---\n\n".join(formatted_chunks)
