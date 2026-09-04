"""Search handler for document search."""

from fastapi import HTTPException
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from pydantic import BaseModel, Field

from vectorize_for_ai.config import settings
from vectorize_for_ai.embeddings import EmbeddingHandler
from vectorize_for_ai.ingestion import DocumentIngestionPipeline
from vectorize_for_ai.logger import get_logger

logger = get_logger(__name__)

# Operator mapping for filter conversion
OPERATOR_MAP = {
    "==": FilterOperator.EQ,
    "EQ": FilterOperator.EQ,
    "!=": FilterOperator.NE,
    "NE": FilterOperator.NE,
    ">": FilterOperator.GT,
    "GT": FilterOperator.GT,
    "<": FilterOperator.LT,
    "LT": FilterOperator.LT,
    ">=": FilterOperator.GTE,
    "GTE": FilterOperator.GTE,
    "<=": FilterOperator.LTE,
    "LTE": FilterOperator.LTE,
    "IN": FilterOperator.IN,
    "NIN": FilterOperator.NIN,
}


class FilterItem(BaseModel):
    """Individual filter specification."""

    key: str = Field(..., description="Metadata field to filter on")
    value: str | int | float | list[str] = Field(..., description="Value to compare against")
    operator: str = Field(
        default="==",
        description="Comparison operator (==, !=, >, <, >=, <=, IN, NIN)",
    )


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Search query text", min_length=1)
    limit: int = Field(
        default=10,
        description="Number of results to return",
        ge=1,
        le=100,
    )
    similarity_top_k: int = Field(
        default=10,
        description="Number of top similar results to retrieve",
        ge=1,
        le=100,
    )
    filters: list[FilterItem] | None = Field(
        default=None,
        description="Optional metadata filters to apply (combined with AND logic)",
    )


class SearchResult(BaseModel):
    """Search result model."""

    id: str = Field(..., description="Node ID")
    text: str = Field(..., description="Document text content")
    score: float = Field(..., description="Relevance score")
    metadata: dict = Field(default_factory=dict, description="Node metadata")


class SearchResponse(BaseModel):
    """Search response model."""

    query: str = Field(default="", description="Original search query")
    results: list[SearchResult] = Field(default=[], description="Search results")
    total: int = Field(default=0, description="Total number of results returned")


def _build_metadata_filters(
    filter_items: list[FilterItem] | None,
) -> MetadataFilters | None:
    """Convert filter items to LlamaIndex MetadataFilters.

    Args:
        filter_items: List of filter specifications

    Returns:
        MetadataFilters object or None if no filters provided

    Raises:
        HTTPException: If invalid operator is provided
    """
    if not filter_items:
        return None

    metadata_filters = []
    for item in filter_items:
        operator = OPERATOR_MAP.get(item.operator.upper())
        if operator is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filter operator: {item.operator}. "
                f"Supported operators: {', '.join(OPERATOR_MAP.keys())}",
            )

        metadata_filters.append(
            MetadataFilter(
                key=item.key,
                value=item.value,
                operator=operator,
            )
        )

    return MetadataFilters(filters=metadata_filters)


async def search(
    request: SearchRequest,
    ingestion_pipeline: DocumentIngestionPipeline | None,
    embedding_handler: EmbeddingHandler | None,
) -> SearchResponse:
    """Search for documents based on query."""
    if not ingestion_pipeline or not embedding_handler:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized",
        )

    try:
        await logger.ainfo(f"Processing search query: {request.query}")

        # Build metadata filters if provided
        metadata_filters = _build_metadata_filters(request.filters)
        if metadata_filters:
            await logger.ainfo("Applying %s metadata filter(s)", request.filters)

        # Generate dense embedding for query
        query_embedding = embedding_handler.get_dense_embedding(request.query)

        # Create VectorStoreQuery - use HYBRID mode for Milvus, DEFAULT for OpenSearch
        # OpenSearch requires lexical_query or search_pipeline for hybrid search
        query_mode = (
            VectorStoreQueryMode.HYBRID
            if settings.database_type == "milvus"
            else VectorStoreQueryMode.DEFAULT
        )

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=request.query,
            sparse_top_k=request.similarity_top_k,
            similarity_top_k=request.similarity_top_k,
            mode=query_mode,
            filters=metadata_filters,
        )

        # Perform search using LlamaIndex vector store
        query_result = ingestion_pipeline.db_handler.query(vector_store_query)

        # Format response
        search_results = []
        for node, score in zip(query_result.nodes, query_result.similarities or []):
            node.metadata.pop("text", None)
            search_results.append(
                SearchResult(
                    id=node.node_id,
                    text=node.get_content(),
                    score=score if score is not None else 0.0,
                    metadata=node.metadata,
                )
            )

        # Limit results to requested amount
        search_results = search_results[: request.limit]

        await logger.ainfo(f"Found {len(search_results)} results for query: {request.query}")

        return SearchResponse(
            query=request.query,
            results=search_results,
            total=len(search_results),
        )

    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )
