"""Background worker for Google Drive and local document ingestion into OpenSearch."""

import asyncio
import json
import mimetypes
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path

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


async def run_local_ingestion_job(
    job_id: str,
    directory_path: str,
    start_date: str,
    redis_client,
    embedding_handler: EmbeddingHandler,
    pipeline: DocumentIngestionPipeline,
) -> None:
    """
    Background coroutine. Scans local directory for files modified >= start_date,
    generates dense + sparse embeddings, and indexes them into the vector store.
    Progress and final status are persisted in Redis.
    """
    try:
        submitted_at = datetime.now(timezone.utc).isoformat()
        await _set_job(redis_client, job_id, {
            "job_id": job_id,
            "status": JobStatus.RUNNING,
            "start_date": start_date,
            "directory_path": directory_path,
            "submitted_at": submitted_at,
            "documents_processed": 0,
            "nodes_indexed": 0,
            "error": None,
        })

        target_date = date.fromisoformat(start_date)
        base_dir = Path(directory_path).expanduser().resolve()

        def _process_local_files() -> tuple[int, int]:
            p_docs = 0
            p_nodes = 0

            for file_path in base_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                # Skip hidden files or files within hidden directories
                if any(part.startswith(".") for part in file_path.relative_to(base_dir).parts):
                    continue

                try:
                    stat = file_path.stat()
                    file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).date()
                    if file_mtime < target_date:
                        continue

                    content = file_path.read_bytes()
                    mime_type, _ = mimetypes.guess_type(str(file_path))
                    mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

                    metadata = {
                        "name": file_path.name,
                        "mimeType": mime_type or "application/octet-stream",
                        "id": str(file_path),
                        "createdTime": mtime_iso,
                        "modifiedTime": mtime_iso,
                        "size": stat.st_size,
                        "local_path": str(file_path),
                    }

                    count = pipeline.ingest_document(content, metadata, embedding_handler)
                    p_nodes += count
                    p_docs += 1
                    logger.info("Ingested local file '%s' (%d nodes)", file_path.name, count)
                except Exception as file_err:
                    logger.warning("Failed to process local file '%s': %s", file_path, file_err)

            return p_docs, p_nodes

        loop = asyncio.get_event_loop()
        docs_processed, nodes_indexed = await loop.run_in_executor(None, _process_local_files)

        await _set_job(redis_client, job_id, {
            "job_id": job_id,
            "status": JobStatus.COMPLETED,
            "start_date": start_date,
            "directory_path": directory_path,
            "submitted_at": submitted_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "documents_processed": docs_processed,
            "nodes_indexed": nodes_indexed,
            "error": None,
        })
        logger.info(
            "Local ingestion job %s completed: %d documents, %d nodes",
            job_id, docs_processed, nodes_indexed,
        )

    except Exception as exc:
        logger.error("Local ingestion job %s failed: %s", job_id, exc)
        existing = await _get_job(redis_client, job_id) or {}
        await _set_job(redis_client, job_id, {
            **existing,
            "job_id": job_id,
            "status": JobStatus.FAILED,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
