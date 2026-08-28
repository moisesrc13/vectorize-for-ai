import uuid
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated

import fakeredis
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from vectorize_for_ai.config import settings
from vectorize_for_ai.embeddings import EmbeddingHandler
from vectorize_for_ai.gdrive_ingest_worker import JobStatus, _get_job, run_ingestion_job
from vectorize_for_ai.ingestion import DocumentIngestionPipeline
from vectorize_for_ai.logger import get_logger
from vectorize_for_ai.search_handler import SearchRequest, SearchResponse, search

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Global application-level singletons (initialised in lifespan)
# ---------------------------------------------------------------------------

embedding_handler: EmbeddingHandler | None = None
ingestion_pipeline: DocumentIngestionPipeline | None = None
redis_client: fakeredis.FakeRedis | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global embedding_handler, ingestion_pipeline, redis_client

    logger.info("Starting up FastAPI application")
    try:
        logger.info("Initializing 📃️ document ingestion pipeline")
        ingestion_pipeline = DocumentIngestionPipeline(settings.chunk_max_tokens)
        embedding_handler = EmbeddingHandler()
        redis_client = fakeredis.FakeRedis(decode_responses=True)
        logger.info(
            "Initialized %s, embedding handlers, and in-process Redis store",
            settings.database_type,
        )
    except Exception as e:
        logger.error("Failed to initialize handlers: %s", e)
        raise

    yield

    # Shutdown
    if ingestion_pipeline:
        ingestion_pipeline.close()
    if redis_client:
        redis_client.close()
    logger.info("Shutting down FastAPI application")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

api_key = APIKeyHeader(name="api-key")


def authenticate(x_api_key: str = Depends(api_key)) -> str:
    if x_api_key != settings.api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )
    return x_api_key


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        f"API for vectorize and search documents using {settings.database_type} "
        "with LlamaIndex"
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper: dependency accessors
# ---------------------------------------------------------------------------

def _get_redis() -> fakeredis.FakeRedis:
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return redis_client


def _get_embedding_handler() -> EmbeddingHandler:
    if embedding_handler is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return embedding_handler


# ---------------------------------------------------------------------------
# Pydantic models for /ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Request body for the ingestion trigger endpoint."""

    start_date: date = Field(
        ...,
        description="Ingest documents with createdTime >= this date (YYYY-MM-DD)",
        examples=["2024-01-01"],
    )


class IngestResponse(BaseModel):
    """Response returned immediately after ingestion job is submitted."""

    job_id: str = Field(..., description="UUID of the background ingestion job")
    status: JobStatus = Field(..., description="Initial job status")
    start_date: str = Field(..., description="Start date used for filtering")
    message: str = Field(..., description="Human-readable confirmation")


class JobStatusResponse(BaseModel):
    """Job status as stored in Redis."""

    job_id: str
    status: JobStatus
    start_date: str | None = None
    submitted_at: str | None = None
    completed_at: str | None = None
    documents_processed: int = 0
    nodes_indexed: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root_endpoint() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Document Search API",
        "version": settings.api_version,
        "database": settings.database_type,
        "status": "running",
    }


@app.get("/health")
async def health_endpoint() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "database": settings.database_type}


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest,
    _: Annotated[None, Depends(authenticate)],
) -> SearchResponse:
    """Search for documents based on query."""
    return await search(request, ingestion_pipeline, embedding_handler)


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_endpoint(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(authenticate)],
    r: Annotated[fakeredis.FakeRedis, Depends(_get_redis)],
    emb: Annotated[EmbeddingHandler, Depends(_get_embedding_handler)],
) -> IngestResponse:
    """
    Submit a background ingestion job.

    Streams all Google Drive documents whose **createdTime** is >= *start_date*,
    generates dense (HuggingFace) **and** sparse (BM25) embeddings for each chunk,
    and indexes them into the OpenSearch vector store for hybrid search.

    Returns a job UUID immediately.  Poll ``GET /ingest/job?job_id=<uuid>`` for status.
    """
    job_id = str(uuid.uuid4())
    start_date_str = request.start_date.isoformat()

    await logger.ainfo(
        "Submitting ingestion job %s for documents >= %s", job_id, start_date_str
    )

    background_tasks.add_task(
        run_ingestion_job,
        job_id=job_id,
        start_date=start_date_str,
        redis_client=r,
        embedding_handler=emb,
    )

    return IngestResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        start_date=start_date_str,
        message=f"Ingestion job submitted. Poll /ingest/job?job_id={job_id} for status.",
    )


@app.get("/ingest/job", response_model=JobStatusResponse)
async def ingest_job_status_endpoint(
    job_id: str,
    _: Annotated[None, Depends(authenticate)],
    r: Annotated[fakeredis.FakeRedis, Depends(_get_redis)],
) -> JobStatusResponse:
    """
    Retrieve the current status of a previously submitted ingestion job.

    - **job_id**: UUID returned by ``POST /ingest``.

    Job records expire automatically after 1 hour (TTL).
    """
    job_data = _get_job(r, job_id)
    if job_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found or has expired.",
        )
    return JobStatusResponse(**job_data)
