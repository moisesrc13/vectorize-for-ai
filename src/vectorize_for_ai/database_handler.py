"""Abstract base class for database handlers using LlamaIndex."""

from abc import ABC, abstractmethod
from typing import Any

from kb_doc_db.logger import get_logger
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import VectorStoreQuery

logger = get_logger(__name__)


class DatabaseHandler(ABC):
    """Abstract base class for database operations using LlamaIndex vector stores."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialize database handler."""
        pass

    @abstractmethod
    def get_vector_store(self) -> Any:
        """Get the underlying LlamaIndex vector store instance.

        Returns:
            VectorStore instance (MilvusVectorStore or OpensearchVectorStore)
        """
        pass

    @abstractmethod
    def add_nodes(self, nodes: list[BaseNode]) -> list[str]:
        """Add nodes to the vector store.

        Args:
            nodes: List of LlamaIndex nodes to add

        Returns:
            List of node IDs that were added
        """
        pass

    @abstractmethod
    def query(
        self,
        query: VectorStoreQuery,
    ) -> Any:
        """Query the vector store.

        Args:
            query: VectorStoreQuery object with query parameters

        Returns:
            VectorStoreQueryResult with matching nodes and scores
        """
        pass

    @abstractmethod
    def delete_nodes_by_system_number(self, ai_system_number: str) -> int:
        """Delete all nodes matching the given ai_system_id number.

        Args:
            ai_system_id: The AI system identifier whose nodes should be deleted
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass
