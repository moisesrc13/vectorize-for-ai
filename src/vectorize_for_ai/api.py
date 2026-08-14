from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader

from vectorize_for_ai.config import settings
from vectorize_for_ai.embeddings import EmbeddingHandler
from vectorize_for_ai.ingestion import DocumentIngestionPipeline
from vectorize_for_ai.logger import get_logger
from vectorize_for_ai.search_handler import SearchRequest, SearchResponse, search

logger = get_logger(__name__)

async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Document Search API",
        "version": settings.api_version,
        "database": settings.database_type,
        "status": "running",
    }


async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "database": settings.database_type}

# Global instances
embedding_handler: EmbeddingHandler | None = None
ingestion_pipeline: DocumentIngestionPipeline | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global embedding_handler, ingestion_pipeline, snow_service
    logger.info("Starting up FastAPI application")
    try:
        logger.info("Initializing 📃️ document ingestion pipeline")
        ingestion_pipeline = DocumentIngestionPipeline(settings.chunk_max_tokens)
        embedding_handler = EmbeddingHandler()
        logger.info(f"Initialized {settings.database_type} and embedding handlers")

    except Exception as e:
        logger.error(f"Failed to initialize handlers: {e}")
        raise

    yield

    # Shutdown
    if ingestion_pipeline:
        ingestion_pipeline.close()
    logger.info("Shutting down FastAPI application")


api_key = APIKeyHeader(name="api-key")


def authenticate(x_api_key: str = Depends(api_key)) -> str:
    if x_api_key != settings.api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )
    return x_api_key


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=f"API for vectorize and search documents using {settings.database_type} with LlamaIndex",
    lifespan=lifespan,
)


@app.get("/")
async def root_endpoint() -> dict[str, str]:
    """Root endpoint."""
    return await root()


@app.get("/health")
async def health_endpoint() -> dict[str, str]:
    """Health check endpoint."""
    return await health()



@app.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest, _: Annotated[None, Depends(authenticate)]
) -> SearchResponse:
    """Search for documents based on query."""
    return await search(request, ingestion_pipeline, embedding_handler)
