import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_./:-]+")


class SQLiteLexicalStore:
    """SQLite FTS5-backed lexical index for chunk retrieval."""

    def __init__(
        self,
        db_path: Path = Path("data/indexes/lexical.sqlite"),
    ) -> None:
        self.db_path: Path = db_path

    def rebuild_from_chunks(
        self,
        chunks_path: Path = Path("data/processed/chunks.jsonl"),
    ) -> None:
        """Rebuild the lexical index from persisted chunks."""
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        chunks: list[dict[str, Any]] = self._load_chunks(chunks_path)

        with self._connect() as connection:
            self._drop_tables(connection)
            self._create_tables(connection)
            self._insert_chunks(connection, chunks)

        logger.info(
            "Rebuilt SQLite lexical index at %s from %s chunks",
            self.db_path,
            len(chunks),
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search indexed chunks using SQLite FTS5 BM25 ranking."""
        filters = filters or {}

        if not query.strip():
            return []

        match_query = self._build_match_query(query)

        if not match_query:
            return []

        where_clauses = ["chunk_fts MATCH ?"]
        params: list[Any] = [match_query]

        for key, value in filters.items():
            where_clauses.append(f"chunks.{key} = ?")
            params.append(value)

        params.append(limit)

        sql = f"""
            SELECT
                chunks.chunk_id,
                chunks.source,
                chunks.source_path,
                chunks.content_type,
                chunks.domain,
                chunks.doc_role,
                chunks.document_id,
                chunks.section_id,
                chunks.text,
                bm25(chunk_fts) AS raw_score
            FROM chunk_fts
            JOIN chunks ON chunks.chunk_id = chunk_fts.chunk_id
            WHERE {" AND ".join(where_clauses)}
            ORDER BY raw_score ASC
            LIMIT ?
        """

        with self._connect() as connection:
            rows: list[Any] = connection.execute(sql, params).fetchall()

        return [self._row_to_result(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection."""
        connection: sqlite3.Connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
        """Load and normalize chunks from JSONL."""
        chunks: list[dict[str, Any]] = []

        with chunks_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                raw_chunk = json.loads(line)
                normalized_chunk = SQLiteLexicalStore._normalize_chunk(raw_chunk)

                if not normalized_chunk["chunk_id"]:
                    raise ValueError(
                        f"Chunk on line {line_number} is missing chunk_id: {raw_chunk}"
                    )

                chunks.append(normalized_chunk)

        return chunks

    @staticmethod
    def _drop_tables(connection: sqlite3.Connection) -> None:
        """Drop existing lexical index tables."""
        connection.execute("DROP TABLE IF EXISTS chunk_fts")
        connection.execute("DROP TABLE IF EXISTS chunks")

    @staticmethod
    def _create_tables(connection: sqlite3.Connection) -> None:
        """Create metadata and full-text search tables."""
        connection.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                source TEXT,
                source_path TEXT,
                content_type TEXT,
                domain TEXT,
                doc_role TEXT,
                document_id TEXT,
                section_id TEXT,
                text TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                chunk_id UNINDEXED,
                searchable_text
            )
            """
        )

        connection.execute("CREATE INDEX idx_chunks_source ON chunks(source)")
        connection.execute(
            "CREATE INDEX idx_chunks_content_type ON chunks(content_type)"
        )
        connection.execute("CREATE INDEX idx_chunks_domain ON chunks(domain)")
        connection.execute("CREATE INDEX idx_chunks_doc_role ON chunks(doc_role)")

    @staticmethod
    def _normalize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
        """Normalize chunk JSONL records into the retrieval payload shape."""
        metadata = chunk.get("metadata") or {}

        return {
            "chunk_id": chunk.get("chunk_id")
            or chunk.get("id")
            or metadata.get("chunk_id"),
            "source": chunk.get("source") or metadata.get("source"),
            "source_path": chunk.get("source_path") or metadata.get("source_path"),
            "content_type": chunk.get("content_type") or metadata.get("content_type"),
            "domain": chunk.get("domain") or metadata.get("domain"),
            "doc_role": chunk.get("doc_role") or metadata.get("doc_role"),
            "document_id": chunk.get("document_id") or metadata.get("document_id"),
            "section_id": chunk.get("section_id") or metadata.get("section_id"),
            "text": chunk.get("text") or "",
        }

    def _insert_chunks(
        self,
        connection: sqlite3.Connection,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Insert chunks into metadata and FTS tables."""
        for chunk in chunks:
            self._insert_chunk(connection, chunk)

    def _insert_chunk(
        self,
        connection: sqlite3.Connection,
        chunk: dict[str, Any],
    ) -> None:
        """Insert one chunk into SQLite."""
        chunk_id = chunk["chunk_id"]

        connection.execute(
            """
            INSERT INTO chunks (
                chunk_id,
                source,
                source_path,
                content_type,
                domain,
                doc_role,
                document_id,
                section_id,
                text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                chunk.get("source"),
                chunk.get("source_path"),
                chunk.get("content_type"),
                chunk.get("domain"),
                chunk.get("doc_role"),
                chunk.get("document_id"),
                chunk.get("section_id"),
                chunk.get("text") or "",
            ),
        )

        connection.execute(
            """
            INSERT INTO chunk_fts (
                chunk_id,
                searchable_text
            )
            VALUES (?, ?)
            """,
            (
                chunk_id,
                self._build_searchable_text(chunk),
            ),
        )

    @staticmethod
    def _build_match_query(query: str) -> str:
        """Build a safe SQLite FTS5 MATCH query from user input."""
        tokens: list[Any] = _TOKEN_PATTERN.findall(query)

        if not tokens:
            return ""

        quoted_tokens: list[str] = [
            SQLiteLexicalStore._quote_fts_token(token) for token in tokens
        ]

        return " OR ".join(quoted_tokens)

    @staticmethod
    def _quote_fts_token(token: str) -> str:
        """Quote one FTS token/phrase for SQLite MATCH syntax."""
        escaped_token = token.replace('"', '""')
        return f'"{escaped_token}"'

    @staticmethod
    def _build_searchable_text(chunk: dict[str, Any]) -> str:
        """Build the text indexed by SQLite FTS."""
        return " ".join(
            [
                chunk.get("source") or "",
                chunk.get("source_path") or "",
                chunk.get("chunk_id") or "",
                chunk.get("document_id") or "",
                chunk.get("section_id") or "",
                chunk.get("content_type") or "",
                chunk.get("domain") or "",
                chunk.get("doc_role") or "",
                chunk.get("text") or "",
            ]
        )

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a SQLite row into retrieval result shape."""
        raw_score = row["raw_score"]

        return {
            "score": -raw_score,
            "retrieval_mode": "lexical_sqlite",
            "payload": {
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "source_path": row["source_path"],
                "content_type": row["content_type"],
                "domain": row["domain"],
                "doc_role": row["doc_role"],
                "document_id": row["document_id"],
                "section_id": row["section_id"],
                "text": row["text"],
            },
        }
