"""Background worker for Google Drive document ingestion into OpenSearch."""

import asyncio
import json
from datetime import datetime, timezone
from enum import Enum

from vectorize_for_ai.config import gdrive_settings
from vectorize_for_ai.embeddings import EmbeddingHandler
from vectorize_for_ai.gdrive_document_processor import GDriveDocumentProcessor
from vectorize_for_ai.ingestion import DocumentIngestionPipeline
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


async def _set_job(redis_client, job_id: str, data: dict) -> None:
    await redis_client.set(
        _build_job_key(job_id),
        json.dumps(data),
        ex=JOB_TTL_SECONDS,
    )


async def _get_job(redis_client, job_id: str) -> dict | None:
    raw = await redis_client.get(_build_job_key(job_id))
    if raw is None:
        return None
    return json.loads(raw)


async def run_ingestion_job(
    job_id: str,
    start_date: str,
    redis_client,
    embedding_handler: EmbeddingHandler,
    pipeline: DocumentIngestionPipeline,
) -> None:
    """
    Background coroutine.  Streams Google Drive documents whose createdTime >=
    start_date, generates dense + sparse embeddings, and indexes them into the
    OpenSearch vector store.  Progress and final status are persisted in Redis.
    """
    try:
        submitted_at = datetime.now(timezone.utc).isoformat()
        await _set_job(redis_client, job_id, {
            "job_id": job_id,
            "status": JobStatus.RUNNING,
            "start_date": start_date,
            "submitted_at": submitted_at,
            "documents_processed": 0,
            "nodes_indexed": 0,
            "error": None,
        })

        processor = GDriveDocumentProcessor()

        docs_processed = 0
        nodes_indexed = 0

        def _handler(content: bytes, metadata: dict) -> None:
            nonlocal nodes_indexed
            count = pipeline.ingest_document(content, metadata, embedding_handler)
            nodes_indexed += count

        # stream_new_documents is a synchronous generator — run it in a thread
        # pool so it doesn't block the async event loop.
        loop = asyncio.get_event_loop()

        folder_id = gdrive_settings.drive_folder_id or None
        drive_id = gdrive_settings.drive_shared_id or None

        def _stream() -> list[dict]:
            results = []
            for doc_meta in processor.stream_new_documents(
                drive_id=drive_id,
                folder_id=folder_id,
                since_date=start_date,
                handler=_handler,
                force=True,  # explicit re-ingest: bypass processed-IDs dedup
            ):
                results.append(doc_meta)
            return results

        docs = await loop.run_in_executor(None, _stream)
        docs_processed = len(docs)

        await _set_job(redis_client, job_id, {
            "job_id": job_id,
            "status": JobStatus.COMPLETED,
            "start_date": start_date,
            "submitted_at": submitted_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
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
        existing = await _get_job(redis_client, job_id) or {}
        await _set_job(redis_client, job_id, {
            **existing,
            "job_id": job_id,
            "status": JobStatus.FAILED,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
