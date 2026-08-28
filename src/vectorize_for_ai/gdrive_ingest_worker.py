"""Background worker for Google Drive document ingestion into OpenSearch."""

import asyncio
import json
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path

from llama_index.core.schema import TextNode

from vectorize_for_ai.database_factory import DatabaseFactory
from vectorize_for_ai.embeddings import EmbeddingHandler
from vectorize_for_ai.gdrive_document_processor import GDriveDocumentProcessor
from vectorize_for_ai.logger import get_logger

logger = get_logger(__name__)

JOB_TTL_SECONDS = 3600  # 1 hour


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _build_job_key(job_id: str) -> str:
    return f"ingest:job:{job_id}"


def _set_job(redis_client, job_id: str, data: dict) -> None:
    redis_client.set(
        _build_job_key(job_id),
        json.dumps(data),
        ex=JOB_TTL_SECONDS,
    )


def _get_job(redis_client, job_id: str) -> dict | None:
    raw = redis_client.get(_build_job_key(job_id))
    if raw is None:
        return None
    return json.loads(raw)


def _ingest_document(
    content: bytes,
    metadata: dict,
    embedding_handler: EmbeddingHandler,
    db_handler,
) -> int:
    """
    Convert raw bytes + Google Drive metadata into TextNode(s) with dense + sparse
    embeddings and upsert into the OpenSearch vector index.

    Returns the number of nodes indexed.
    """
    file_name: str = metadata.get("name", "unknown")
    mime_type: str = metadata.get("mimeType", "")

    # Map Google export mime types to a useful extension for the temp file
    _MIME_TO_EXT = {
        "text/plain": ".txt",
        "text/csv": ".csv",
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "image/png": ".png",
        "image/jpeg": ".jpg",
    }
    suffix = _MIME_TO_EXT.get(mime_type, Path(file_name).suffix or ".bin")

    # --- Try Docling parsing for structured formats ---
    nodes: list[TextNode] = []
    try:
        from docling.datamodel.base_models import FormatToExtensions, InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
            PowerpointFormatOption,
            WordFormatOption,
        )
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from llama_index.node_parser.docling import DoclingNodeParser
        from llama_index.readers.docling import DoclingReader

        ext = suffix.lstrip(".")
        # Check if format is supported by Docling
        supported = False
        for fmt in InputFormat:
            exts = FormatToExtensions.get(fmt) or []
            if ext in exts or fmt.value == ext:
                supported = True
                break

        if supported:
            chunker = HybridChunker(max_tokens=512, merge_peers=True)
            node_parser = DoclingNodeParser(chunker=chunker)

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
                documents = reader.load_data(file_path=tmp_path)
                if documents:
                    for doc in documents:
                        doc.metadata.update({
                            "file_name": file_name,
                            "gdrive_id": metadata.get("id", ""),
                            "created_time": metadata.get("createdTime", ""),
                            "mime_type": mime_type,
                        })
                    nodes = node_parser.get_nodes_from_documents(documents)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    except Exception as parse_err:
        logger.warning("Docling parsing failed for %s, falling back to plain-text node: %s", file_name, parse_err)

    # Fallback: single TextNode from raw text content
    if not nodes:
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception:
            text = repr(content[:500])
        nodes = [
            TextNode(
                text=text,
                metadata={
                    "file_name": file_name,
                    "gdrive_id": metadata.get("id", ""),
                    "created_time": metadata.get("createdTime", ""),
                    "mime_type": mime_type,
                },
            )
        ]

    if not nodes:
        logger.warning("No nodes extracted from %s", file_name)
        return 0

    # Remove 'origin' metadata to avoid OpenSearch numeric-overflow issues
    for node in nodes:
        node.metadata.pop("origin", None)

    # --- Dense embeddings ---
    texts = [node.get_content() for node in nodes]
    dense_embeddings = embedding_handler.get_dense_embeddings(texts)
    for node, dense_emb in zip(nodes, dense_embeddings):
        node.embedding = dense_emb

    # --- Sparse embeddings (BM25) stored as extra metadata for hybrid search ---
    sparse_embeddings = embedding_handler.get_sparse_embeddings(texts)
    for node, sparse_emb in zip(nodes, sparse_embeddings):
        # Serialise as JSON string so OpenSearch can store it; search layer can
        # deserialise when building a hybrid query.
        node.metadata["sparse_embedding"] = json.dumps(sparse_emb)

    # --- Upsert into OpenSearch ---
    node_ids = db_handler.add_nodes(nodes)
    logger.info("Indexed %d nodes for '%s'", len(node_ids), file_name)
    return len(node_ids)


async def run_ingestion_job(
    job_id: str,
    start_date: str,
    redis_client,
    embedding_handler: EmbeddingHandler,
) -> None:
    """
    Background coroutine.  Streams Google Drive documents whose createdTime >=
    start_date, generates dense + sparse embeddings, and indexes them into the
    OpenSearch vector store.  Progress and final status are persisted in Redis.
    """
    db_handler = None
    try:
        _set_job(redis_client, job_id, {
            "job_id": job_id,
            "status": JobStatus.RUNNING,
            "start_date": start_date,
            "submitted_at": datetime.utcnow().isoformat(),
            "documents_processed": 0,
            "nodes_indexed": 0,
            "error": None,
        })

        db_handler = DatabaseFactory.create_handler()
        processor = GDriveDocumentProcessor()

        docs_processed = 0
        nodes_indexed = 0

        def _handler(content: bytes, metadata: dict) -> None:
            nonlocal nodes_indexed
            count = _ingest_document(content, metadata, embedding_handler, db_handler)
            nodes_indexed += count

        # stream_new_documents is a synchronous generator — run it in a thread
        # pool so it doesn't block the async event loop.
        loop = asyncio.get_event_loop()

        def _stream() -> list[dict]:
            results = []
            for doc_meta in processor.stream_new_documents(
                since_date=start_date,
                handler=_handler,
            ):
                results.append(doc_meta)
            return results

        docs = await loop.run_in_executor(None, _stream)
        docs_processed = len(docs)

        _set_job(redis_client, job_id, {
            "job_id": job_id,
            "status": JobStatus.COMPLETED,
            "start_date": start_date,
            "submitted_at": _get_job(redis_client, job_id).get("submitted_at"),
            "completed_at": datetime.utcnow().isoformat(),
            "documents_processed": docs_processed,
            "nodes_indexed": nodes_indexed,
            "error": None,
        })
        logger.info(
            "Ingestion job %s completed: %d documents, %d nodes",
            job_id, docs_processed, nodes_indexed,
        )

    except Exception as exc:
        logger.error("Ingestion job %s failed: %s", job_id, exc)
        existing = _get_job(redis_client, job_id) or {}
        _set_job(redis_client, job_id, {
            **existing,
            "job_id": job_id,
            "status": JobStatus.FAILED,
            "completed_at": datetime.utcnow().isoformat(),
            "error": str(exc),
        })
    finally:
        if db_handler:
            try:
                db_handler.close()
            except Exception:
                pass
