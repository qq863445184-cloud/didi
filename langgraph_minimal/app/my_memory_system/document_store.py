from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from time import time
from typing import Any


class SQLiteDocumentStore:
    """SQLite metadata store for RAG documents and chunks.

    Qdrant 适合做向量相似度检索，但文档结构、chunk 正文、来源路径和解析器
    这类元数据更适合放在关系型表里。这个 store 让第八章 RAG 架构里的
    “文档库”和“向量库”分工更清楚。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def upsert_document(
        self,
        *,
        document_id: str,
        namespace: str = "default",
        source_path: str | None = None,
        parser: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rag_documents (
                    document_id, namespace, source_path, parser, metadata_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, namespace) DO UPDATE SET
                    source_path=excluded.source_path,
                    parser=excluded.parser,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    document_id,
                    namespace,
                    source_path,
                    parser,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def replace_chunks(
        self,
        *,
        document_id: str,
        namespace: str = "default",
        chunks: list[dict[str, Any]],
    ) -> None:
        now = time()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM rag_chunks WHERE document_id = ? AND namespace = ?",
                (document_id, namespace),
            )
            conn.executemany(
                """
                INSERT INTO rag_chunks (
                    chunk_id, document_id, namespace, chunk_index, content,
                    section_title, start_char, end_char, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(chunk.get("chunk_id") or f"{document_id}#chunk-{index}"),
                        document_id,
                        namespace,
                        int(chunk.get("chunk_index", index)),
                        str(chunk.get("content", "")),
                        chunk.get("section_title"),
                        chunk.get("start_char"),
                        chunk.get("end_char"),
                        json.dumps(chunk.get("metadata") or {}, ensure_ascii=False),
                        now,
                    )
                    for index, chunk in enumerate(chunks)
                ],
            )

    def get_document(self, document_id: str, *, namespace: str = "default") -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT document_id, namespace, source_path, parser, metadata_json,
                       created_at, updated_at
                FROM rag_documents
                WHERE document_id = ? AND namespace = ?
                """,
                (document_id, namespace),
            ).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self, *, namespace: str | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT document_id, namespace, source_path, parser, metadata_json,
                   created_at, updated_at
            FROM rag_documents
        """
        params: tuple[Any, ...] = ()
        if namespace is not None:
            sql += " WHERE namespace = ?"
            params = (namespace,)
        sql += " ORDER BY updated_at DESC, document_id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._document_from_row(row) for row in rows]

    def list_chunks(self, document_id: str, *, namespace: str = "default") -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, document_id, namespace, chunk_index, content,
                       section_title, start_char, end_char, metadata_json, created_at
                FROM rag_chunks
                WHERE document_id = ? AND namespace = ?
                ORDER BY chunk_index ASC
                """,
                (document_id, namespace),
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def delete_document(self, document_id: str, *, namespace: str = "default") -> bool:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM rag_chunks WHERE document_id = ? AND namespace = ?",
                (document_id, namespace),
            )
            cursor = conn.execute(
                "DELETE FROM rag_documents WHERE document_id = ? AND namespace = ?",
                (document_id, namespace),
            )
        return cursor.rowcount > 0

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    document_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    source_path TEXT,
                    parser TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (document_id, namespace)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    section_title TEXT,
                    start_char INTEGER,
                    end_char INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY (document_id, namespace)
                        REFERENCES rag_documents(document_id, namespace)
                        ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_namespace
                ON rag_chunks(document_id, namespace, chunk_index)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _document_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "document_id": row["document_id"],
            "namespace": row["namespace"],
            "source_path": row["source_path"],
            "parser": row["parser"],
            "metadata": self._decode_json(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _chunk_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "namespace": row["namespace"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "section_title": row["section_title"],
            "start_char": row["start_char"],
            "end_char": row["end_char"],
            "metadata": self._decode_json(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    def _decode_json(self, value: str | None) -> dict[str, Any]:
        if not value:
            return {}
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
