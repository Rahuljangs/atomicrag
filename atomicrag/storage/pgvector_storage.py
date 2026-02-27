"""PostgreSQL + pgvector storage adapter for KnowledgeGraph.

Requires: ``pip install atomicrag[pgvector]``
"""
from __future__ import annotations

from typing import Optional

from atomicrag.models.graph import (
    Chunk,
    Entity,
    KnowledgeGraph,
    KnowledgeUnit,
)


class PGVectorStorage:
    """Save and load a ``KnowledgeGraph`` to/from PostgreSQL with pgvector.

    Example::

        storage = PGVectorStorage("postgresql://user:pass@localhost/db")
        storage.save(graph)
        loaded = storage.load()
    """

    def __init__(
        self,
        connection_string: str,
        schema: str = "atomicrag",
    ):
        try:
            from sqlalchemy import create_engine, text as sql_text
            from sqlalchemy.orm import sessionmaker
        except ImportError:
            raise ImportError(
                "SQLAlchemy is required for PGVectorStorage. "
                "Install with: pip install atomicrag[pgvector]"
            )

        self._sql_text = sql_text
        self.schema = schema
        self.engine = create_engine(connection_string, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _get_session(self):
        session = self.SessionLocal()
        session.execute(self._sql_text(f"SET search_path TO {self.schema}, public"))
        return session

    def create_tables(self) -> None:
        """Create the required tables (idempotent)."""
        session = self._get_session()
        try:
            session.execute(self._sql_text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))

            session.execute(self._sql_text("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    doc_id TEXT,
                    index INTEGER,
                    metadata JSONB DEFAULT '{}'
                )
            """))

            session.execute(self._sql_text("""
                CREATE TABLE IF NOT EXISTS knowledge_units (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding FLOAT8[],
                    chunk_id TEXT,
                    entity_ids TEXT[],
                    metadata JSONB DEFAULT '{}'
                )
            """))

            session.execute(self._sql_text("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT,
                    embedding FLOAT8[],
                    metadata JSONB DEFAULT '{}'
                )
            """))

            session.execute(self._sql_text("""
                CREATE TABLE IF NOT EXISTS edges (
                    ku_id TEXT,
                    entity_id TEXT,
                    PRIMARY KEY (ku_id, entity_id)
                )
            """))

            session.commit()
        finally:
            session.close()

    def save(self, graph: KnowledgeGraph) -> None:
        """Write the graph to PostgreSQL."""
        import json
        session = self._get_session()

        try:
            for c in graph.chunks:
                session.execute(self._sql_text("""
                    INSERT INTO chunks (id, content, doc_id, index, metadata)
                    VALUES (:id, :content, :doc_id, :index, :metadata)
                    ON CONFLICT (id) DO NOTHING
                """), {
                    "id": c.id, "content": c.content, "doc_id": c.doc_id,
                    "index": c.index, "metadata": json.dumps(c.metadata),
                })

            for ku in graph.knowledge_units:
                session.execute(self._sql_text("""
                    INSERT INTO knowledge_units (id, content, embedding, chunk_id, entity_ids, metadata)
                    VALUES (:id, :content, :embedding, :chunk_id, :entity_ids, :metadata)
                    ON CONFLICT (id) DO NOTHING
                """), {
                    "id": ku.id, "content": ku.content,
                    "embedding": ku.embedding, "chunk_id": ku.chunk_id,
                    "entity_ids": ku.entity_ids,
                    "metadata": json.dumps(ku.metadata),
                })

            for e in graph.entities:
                session.execute(self._sql_text("""
                    INSERT INTO entities (id, name, entity_type, embedding, metadata)
                    VALUES (:id, :name, :entity_type, :embedding, :metadata)
                    ON CONFLICT (id) DO NOTHING
                """), {
                    "id": e.id, "name": e.name, "entity_type": e.entity_type,
                    "embedding": e.embedding,
                    "metadata": json.dumps(e.metadata),
                })

            for ku_id, entity_id in graph.edges:
                session.execute(self._sql_text("""
                    INSERT INTO edges (ku_id, entity_id)
                    VALUES (:ku_id, :entity_id)
                    ON CONFLICT DO NOTHING
                """), {"ku_id": ku_id, "entity_id": entity_id})

            session.commit()
        finally:
            session.close()

    def load(self) -> KnowledgeGraph:
        """Read the graph from PostgreSQL."""
        import json
        session = self._get_session()

        try:
            chunks = [
                Chunk(
                    id=r[0], content=r[1], doc_id=r[2] or "",
                    index=r[3] or 0, metadata=json.loads(r[4]) if r[4] else {},
                )
                for r in session.execute(
                    self._sql_text("SELECT id, content, doc_id, index, metadata FROM chunks")
                ).fetchall()
            ]

            kus = [
                KnowledgeUnit(
                    id=r[0], content=r[1],
                    embedding=list(r[2]) if r[2] else [],
                    chunk_id=r[3] or "",
                    entity_ids=list(r[4]) if r[4] else [],
                    metadata=json.loads(r[5]) if r[5] else {},
                )
                for r in session.execute(
                    self._sql_text("SELECT id, content, embedding, chunk_id, entity_ids, metadata FROM knowledge_units")
                ).fetchall()
            ]

            entities = [
                Entity(
                    id=r[0], name=r[1], entity_type=r[2] or "",
                    embedding=list(r[3]) if r[3] else [],
                    metadata=json.loads(r[4]) if r[4] else {},
                )
                for r in session.execute(
                    self._sql_text("SELECT id, name, entity_type, embedding, metadata FROM entities")
                ).fetchall()
            ]

            edges = [
                (r[0], r[1])
                for r in session.execute(
                    self._sql_text("SELECT ku_id, entity_id FROM edges")
                ).fetchall()
            ]

            return KnowledgeGraph(
                chunks=chunks, knowledge_units=kus,
                entities=entities, edges=edges,
            )
        finally:
            session.close()
