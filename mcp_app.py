"""MCP server exposing a search tool that calls the vectorize-for-ai search API."""

import asyncio

import httpx
from fastmcp import FastMCP

from vectorize_for_ai.config import settings
from vectorize_for_ai.logger import get_logger
from vectorize_for_ai.search_handler import SearchResponse

logger = get_logger(__name__)
# ---------------------------------------------------------------------------
# Configuration — resolved from environment variables
# ---------------------------------------------------------------------------

mcp = FastMCP("vectorize-search")


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool()
async def search(
    query: str
) -> str:
    """Search for documents in the vector store.

    Args:
        query: The search query text.
        limit: Maximum number of results to return (1-100, default 10).
        similarity_top_k: Number of top similar results to retrieve (1-100, default 10).

    Returns:
        Numbered list of relevant document passages, ready for an LLM to consume.
    """
    if not settings.api_key:
        raise ValueError(
            "SEARCH_API_KEY environment variable is not set. "
            "Set it to the API key required by the search service."
        )

    payload = {
        "query": query,
        "limit": 10,
        "similarity_top_k": 10,
    }

    headers = {
        "accept": "application/json",
        "api-key": settings.api_key.get_secret_value(),
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"http://{settings.api_host}:{settings.api_port}/search", json=payload, headers=headers)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Search API returned HTTP {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    response_data = SearchResponse(**data)

    if not response_data.results:
        return f'no results found for query: "{query}"'

    passages = []
    for i, result in enumerate(response_data.results, start=1):
        source_url = result.metadata.get("gdrive_url") or result.metadata.get("file_name", "")
        chunk_str = (
            f"[Source {i}]\n"
            f"URL: {source_url}\n"
            f"Content:\n{result.text}"
        )
        passages.append(chunk_str)

    return f'Search results for "{query}":\n\n' + "\n\n".join(passages)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(
            mcp.run_async(
                transport="http", host=settings.mcp_host, port=settings.mcp_port
            )
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception:
        logger.exception("server error")
    finally:
        logger.info("server shutdown complete")
