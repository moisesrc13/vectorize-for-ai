import asyncio
from typing import cast

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.vector_stores.opensearch import OpensearchVectorClient, OpensearchVectorStore
from opensearchpy import OpenSearch

from vectorize_for_ai.config import settings
from vectorize_for_ai.database_handler import DatabaseHandler
from vectorize_for_ai.logger import get_logger

logger = get_logger(__name__)


class OpenSearchHandler(DatabaseHandler):
    """Handler for OpenSearch operations using LlamaIndex OpensearchVectorStore."""

    def __init__(self) -> None:
        """Initialize OpenSearch handler with LlamaIndex vector store."""
        self.index_name = settings.opensearch_index_name
        self.embedding_dim = settings.embedding_dimensions

        if not settings.opensearch_use_ssl:
            self.client = OpensearchVectorClient(
                endpoint=settings.opensearch_uri,
                index=self.index_name,
                dim=self.embedding_dim,
                embedding_field=settings.opensearch_embedding_field,
                text_field=settings.opensearch_text_field,
                use_ssl=False,
                verify_certs=False,
                idx_conf=settings.opensearch_index_config,
            )
        else:
            self.client = OpensearchVectorClient(
                endpoint=settings.opensearch_uri,
                index=self.index_name,
                dim=self.embedding_dim,
                embedding_field=settings.opensearch_embedding_field,
                text_field=settings.opensearch_text_field,
                http_auth=(settings.opensearch_user, settings.opensearch_initial_admin_password),
                use_ssl=settings.opensearch_use_ssl,
                verify_certs=settings.opensearch_verify_certs,
                idx_conf=settings.opensearch_index_config,
            )

        self.vector_store = OpensearchVectorStore(self.client)

        # LlamaIndex defers index creation until first use; trigger it eagerly
        # at startup so the index exists before any ingestion or query occurs.
        self.client._ensure_initialized()

        logger.info(
            "Initialized OpensearchVectorStore for index %s at %s",
            self.index_name,
            settings.opensearch_uri,
        )

    def get_vector_store(self) -> OpensearchVectorStore:
        return self.vector_store

    def add_nodes(self, nodes: list[BaseNode]) -> list[str]:
        try:
            # Delete existing nodes with the same file_name before adding new ones
            if nodes:
                file_name = nodes[0].metadata.get("file_name")
                if file_name:
                    self._delete_nodes_by_filename_sync(file_name)

            # Remove 'origin' from metadata to avoid numeric overflow issues with binary_hash
            for node in nodes:
                if "origin" in node.metadata:
                    del node.metadata["origin"]

            logger.info(
                "Adding %s nodes to OpenSearch index %s", len(nodes), self.index_name
            )
            node_ids = self.vector_store.add(nodes)
            logger.info(
                "Added %s nodes to OpenSearch index '%s'", len(nodes), self.index_name
            )
            return node_ids
        except Exception as e:
            logger.error("Failed to add nodes to OpenSearch: %s", e)
            raise


    def _delete_nodes_by_filename_sync(self, file_name: str) -> int:
        """Synchronous helper — used internally by add_nodes."""
        deleted_count = 0
        try:
            logger.info(
                "Deleting existing nodes with file_name '%s' from OpenSearch index '%s'",
                file_name,
                self.index_name,
            )
            query = {"query": {"term": {"metadata.file_name.keyword": file_name}}}
            response = cast(OpenSearch, self.client._os_client).delete_by_query(
                index=self.index_name,
                body=query,
                params={"refresh": "true"},
            )
            deleted_count = response.get("deleted", 0)
            if deleted_count > 0:
                logger.info(
                    "Deleted %s existing nodes with file_name '%s' from OpenSearch",
                    deleted_count,
                    file_name,
                )
        except Exception as e:
            logger.warning(
                "Failed to delete existing nodes for file_name '%s': %s", file_name, e
            )
        return deleted_count

    async def delete_nodes_by_filename(self, file_name: str) -> int:
        """Async public API — offloads the blocking delete_by_query to a thread."""
        return await asyncio.to_thread(self._delete_nodes_by_filename_sync, file_name)

    def query(
        self,
        query: VectorStoreQuery,
    ) -> VectorStoreQueryResult:
        try:
            result = self.vector_store.query(query)
            logger.info("Query returned %s results", len(result.nodes))
            return result
        except Exception as e:
            logger.error("Failed to query OpenSearch: %s", e)
            raise

    def close(self) -> None:
        try:
            self.client.close()
            logger.info("Disconnected from OpenSearch")
        except Exception as e:
            logger.warning("error disconnecting from OpenSearch: %s", e)

    async def aclose(self) -> None:
        try:
            await self.vector_store.aclose()
            logger.info("Disconnected from OpenSearch (async)")
        except Exception as e:
            logger.warning("error disconnecting from OpenSearch (async): %s", e)
