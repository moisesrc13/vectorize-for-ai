"""MCP server exposing a search tool that calls the vectorize-for-ai search API."""

import asyncio

import httpx
from mcp.server.fastmcp import FastMCP

from vectorize_for_ai.config import settings
from vectorize_for_ai.logger import get_logger

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
    query: str,
    limit: int = 10,
    similarity_top_k: int = 10,
) -> dict:
    """Search for documents in the vector store.

    Args:
        query: The search query text.
        limit: Maximum number of results to return (1-100, default 10).
        similarity_top_k: Number of top similar results to retrieve (1-100, default 10).

    Returns:
        A dict containing the original query, a list of results (id, text, score,
        metadata), and the total count of results returned.
    """
    if not settings.api_key:
        raise ValueError(
            "SEARCH_API_KEY environment variable is not set. "
            "Set it to the API key required by the search service."
        )

    payload = {
        "query": query,
        "limit": limit,
        "similarity_top_k": similarity_top_k,
    }

    headers = {
        "accept": "application/json",
        "api-key": settings.api_key,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"http://{settings.api_host}:{settings.api_port}", json=payload, headers=headers)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Search API returned HTTP {resp.status_code}: {resp.text}"
        )

    return resp.json()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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
