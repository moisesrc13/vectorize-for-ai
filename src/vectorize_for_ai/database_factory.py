"""Factory for creating database handler instances."""

from kb_doc_db.config import settings
from kb_doc_db.database_handler import DatabaseHandler
from kb_doc_db.logger import get_logger
from kb_doc_db.milvus_client import MilvusHandler
from kb_doc_db.opensearch_handler import OpenSearchHandler

logger = get_logger(__name__)


class DatabaseFactory:
    """Factory class for creating database handler instances."""

    @staticmethod
    def create_handler() -> DatabaseHandler:
        """Create and return appropriate database handler based on configuration.

        Returns:
            DatabaseHandler instance (MilvusHandler or OpenSearchHandler)

        Raises:
            ValueError: If database_type is not supported
        """
        database_type = settings.database_type.lower()

        if database_type == "milvus":
            logger.info("Creating Milvus database handler")
            return MilvusHandler()
        if database_type == "opensearch":
            logger.info("Creating OpenSearch database handler")
            return OpenSearchHandler()
        raise ValueError(
            f"Unsupported database type: {database_type}. "
            f"Supported types are: 'milvus', 'opensearch'"
        )
