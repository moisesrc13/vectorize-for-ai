"""Factory for creating database handler instances."""

from vectorize_for_ai.config import settings
from vectorize_for_ai.database_handler import DatabaseHandler
from vectorize_for_ai.logger import get_logger
from vectorize_for_ai.opensearch_handler import OpenSearchHandler

logger = get_logger(__name__)


class DatabaseFactory:
    """Factory class for creating database handler instances."""

    @staticmethod
    def create_handler() -> DatabaseHandler:
        """Create and return appropriate database handler based on configuration.

        Returns:
            DatabaseHandler instance (- or OpenSearchHandler)

        Raises:
            ValueError: If database_type is not supported
        """
        database_type = settings.database_type.lower()
        if database_type == "opensearch":
            logger.info("Creating OpenSearch database handler")
            return OpenSearchHandler()
        raise ValueError(
            f"Unsupported database type: {database_type}. "
            f"Supported types are: 'milvus', 'opensearch'"
        )
