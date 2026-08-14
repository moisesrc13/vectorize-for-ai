import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.security import APIKeyHeader

from vectorize_for_ai.base import (
    DeleteNodesBySystemNumberRequest,
    DeleteNodesBySystemNumberResponse,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    SimilarUseCasesResponse,
    UsesCasesReportRequest,
)
from vectorize_for_ai.config import settings
from vectorize_for_ai.embeddings import EmbeddingHandler
from vectorize_for_ai.handlers import (
    get_similar_usecases_report,
    get_similar_usecases_with_context_report,
    get_task_progress,
    health,
    load_from_cos,
    load_from_snow_ai_systems,
    load_from_snow_system_id,
    root,
    search,
    search_similar_mcp_servers,
    sync_mcp_servers,
)
from vectorize_for_ai.handlers.search_handler import SearchRequest, SearchResponse
from vectorize_for_ai.ingestion import DocumentIngestionPipeline
from vectorize_for_ai.logger import get_logger
from vectorize_for_ai.snow_service import SNOWService
from vectorize_for_ai.task_manager import TaskStatus, TaskStatusEnum, set_task_status

logger = get_logger(__name__)

# Global instances
embedding_handler: EmbeddingHandler | None = None
ingestion_pipeline: DocumentIngestionPipeline | None = None
snow_service: SNOWService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global embedding_handler, ingestion_pipeline, snow_service
    logger.info("Starting up FastAPI application")
    try:
        logger.info("Initializing 📃️ document ingestion pipeline")
        ingestion_pipeline = DocumentIngestionPipeline(settings.chunk_max_tokens)
        embedding_handler = EmbeddingHandler()
        snow_service = SNOWService(ingestion_pipeline)
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


@app.put(
    "/load/cos/{start_date}",
    status_code=201,
    description="Trigger a load from COS from a start date YYYY-MM-DD",
)
async def load_cos_endpoint(
    start_date_time: str,
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(authenticate)],
):
    """Load documents from a specific start date."""
    return await load_from_cos(start_date_time, background_tasks, ingestion_pipeline)


@app.put(
    "/load/snow-ai-systems/{start_date}",
    status_code=201,
    description="Trigger a load from ServiceNow from a start date YYYY-MM-DD",
)
async def load_snow_ai_systems_endpoint(
    start_date: str,
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(authenticate)],
):
    """Load documents from ServiceNow AI systems created after a specific start date."""
    return await load_from_snow_ai_systems(
        start_date, background_tasks, ingestion_pipeline
    )


@app.put(
    "/load/snow-ai-system-id/{number}",
    status_code=201,
    description="Trigger a load from ServiceNow by system ID",
)
async def load_snow_system_id_endpoint(
    number: str,
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(authenticate)],
):
    """Load document from a single ServiceNow AI system by number."""
    return await load_from_snow_system_id(number, background_tasks, ingestion_pipeline)


@app.get("/task/{task_id}/progress", response_model=TaskStatus)
async def get_task_progress_endpoint(
    task_id: str, _: Annotated[None, Depends(authenticate)]
):
    """Get the progress of a task."""
    return await get_task_progress(task_id)


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest, _: Annotated[None, Depends(authenticate)]
) -> SearchResponse:
    """Search for documents based on query."""
    return await search(request, ingestion_pipeline, embedding_handler)


@app.post(
    "/similar-usecases/context",
    response_model=SimilarUseCasesResponse,
)
async def similar_usecases_context_endpoint(
    _: Annotated[None, Depends(authenticate)],
    file: UploadFile | None = File(default=None),
    context: str | None = Form(default=None),
) -> SimilarUseCasesResponse:
    """Generate a similar use cases report based on the provided context and file.
    Args:
        file: The file to convert to markdown
        context: AI project context
    Returns:
        SimilarUseCasesResponse with similar use cases information
    """
    report = await get_similar_usecases_with_context_report(
        file=file,
        context=context,
        ingestion_pipeline=ingestion_pipeline,
        embedding_handler=embedding_handler,
    )
    return report


@app.post(
    "/similar-usecases",
    response_model=SimilarUseCasesResponse,
)
async def similar_usecases_endpoint(
    background_tasks: BackgroundTasks,
    request: UsesCasesReportRequest,
    _: Annotated[None, Depends(authenticate)],
) -> SimilarUseCasesResponse:
    """Find similar use cases for a SNOW AI system id using WatsonX.

    Download system id metadata and get attachment using sys id
    Converts the file to markdown first, then uses WatsonX LLM to extract:
    - Project overview
    - Proposed functionality
    - Project benefits
    - General functionality

    Then searches for similar use cases based on the extracted information.
    If ingest=True, the document is ingested into the vector database as a background task.

    Args:
        request: A UsesCasesReportRequest

    Returns:
        SimilarUseCasesResponse with similar use cases information
    """
    report = await get_similar_usecases_report(
        background_tasks=background_tasks,
        request=request,
        snow_service=snow_service,
        ingestion_pipeline=ingestion_pipeline,
        embedding_handler=embedding_handler,
    )
    return report


@app.post(
    "/search-mcp-servers",
    status_code=status.HTTP_200_OK,
)
async def similar_mcp_servers_endpoint(
    request: SimilaritySearchRequest,
    _: Annotated[None, Depends(authenticate)],
) -> SimilaritySearchResponse:
    """Search for similar MCP servers based on text description.

    Uses embeddings stored in Milvus to compare the provided text against MCP server
    descriptions and returns the most similar matches.

    Args:
        request: SimilaritySearchRequest containing:
            - query: Text to compare against MCP server descriptions
            - top_k: Number of top matches to return (default: 5, max: 20)
            - threshold: Minimum similarity score (0-1, default: 0.7)

    Returns:
        SimilaritySearchResponse with list of similar MCP servers, each containing:
        - sys_id: System ID of the MCP server
        - name: Name of the MCP server
        - description: Description of the MCP server
        - similarity_score: Cosine similarity score (0-1, higher is better)
    """
    return await search_similar_mcp_servers(request, embedding_handler)


@app.post(
    "/nodes/delete-by-ai-system",
    status_code=status.HTTP_200_OK,
    response_model=DeleteNodesBySystemNumberResponse,
    description="Delete all nodes for a given ServiceNow AI system number",
)
async def delete_nodes_by_ai_system_endpoint(
    request: DeleteNodesBySystemNumberRequest,
    _: Annotated[None, Depends(authenticate)],
) -> DeleteNodesBySystemNumberResponse:
    """Delete all vector store nodes associated with the given AI system number.

    Args:
        request: Payload containing the AI system number whose nodes should be deleted

    Returns:
        DeleteNodesBySystemNumberResponse with the system number and count of deleted nodes
    """
    if ingestion_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion pipeline not initialized",
        )
    deleted_count = ingestion_pipeline._delete_nodes_by_system_number(
        request.ai_system_number
    )
    return DeleteNodesBySystemNumberResponse(
        ai_system_number=request.ai_system_number,
        deleted_count=deleted_count,
    )


@app.put(
    "/sync-mcp-servers",
    status_code=status.HTTP_202_ACCEPTED,
    description="Sync MCP servers from ServiceNow to Milvus as a background task",
)
async def sync_mcp_servers_endpoint(
    _: Annotated[None, Depends(authenticate)],
) -> dict:
    """Sync MCP servers from ServiceNow CMDB to Milvus.

    Fetches all MCP servers from ServiceNow, generates embeddings, and stores
    them in Milvus for fast similarity search. This endpoint should be called
    periodically to keep the MCP server data up to date.

    Returns 202 immediately and runs the sync in the background. Use the
    returned task_id with the /task/{task_id}/progress endpoint to monitor
    the sync progress.

    Returns:
        Dictionary with:
        - task_id: Unique identifier to track the sync progress
        - message: Status message
    """
    task_id = str(uuid.uuid4())

    set_task_status(
        task_id,
        TaskStatus(
            status=TaskStatusEnum.running,
            progress=0,
            total=100,
            message="MCP server sync task queued",
        ),
    )
    _task = asyncio.create_task(sync_mcp_servers(task_id, embedding_handler))

    return {
        "task_id": task_id,
        "message": f"MCP server sync started in background. Use /task/{task_id}/progress to check status.",
    }
