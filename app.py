"""Entry point for the Vectorize-for-AI API server."""

import uvicorn

from vectorize_for_ai.config import settings
from vectorize_for_ai.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Start the uvicorn ASGI server."""
    # Bind only to localhost; never expose on 0.0.0.0 (all interfaces).
    host = "127.0.0.1"
    if settings.api_host not in ("0.0.0.0", ""):
        # Honour an explicit non-wildcard override from the environment.
        host = settings.api_host

    logger.info(
        "Starting %s v%s on %s:%d",
        settings.api_title,
        settings.api_version,
        host,
        settings.api_port,
    )

    uvicorn.run(
        "vectorize_for_ai.api:app",
        host=host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
