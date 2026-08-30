"""Embedding generation for dense and sparse vectors."""

# torch._dynamo is disabled globally in vectorize_for_ai/__init__.py
# before any model is imported. No need to repeat it here.

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pymilvus.model.sparse import BM25EmbeddingFunction

from vectorize_for_ai.config import settings
from vectorize_for_ai.logger import get_logger

logger = get_logger(__name__)

class EmbeddingHandler:
    """Handler for generating dense and sparse embeddings."""

    def __init__(self) -> None:
        """Initialize embedding models."""
        self.dense_model = self._initialize_dense_model()
        self.sparse_model = self._initialize_sparse_model()

    def _initialize_dense_model(self) -> HuggingFaceEmbedding:
        """Initialize HuggingFace dense embedding model.

        Returns:
            Initialized HuggingFace embedding model
        """
        try:
            model = HuggingFaceEmbedding(
                model_name=settings.embedding_model_name,
                embed_batch_size=32,
            )
            logger.info(f"Initialized dense embedding model: {settings.embedding_model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize dense embedding model: {e}")
            raise

    def _initialize_sparse_model(self) -> BM25EmbeddingFunction:
        """Initialize BM25 sparse embedding model.

        Returns:
            Initialized BM25 embedding function
        """
        try:
            model = BM25EmbeddingFunction()
            logger.info("Initialized BM25 sparse embedding model")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize sparse embedding model: {e}")
            raise

    def get_dense_embedding(self, text: str) -> list[float]:
        """Generate dense embedding for text.

        Args:
            text: Input text

        Returns:
            Dense embedding vector
        """
        try:
            embedding = self.dense_model.get_text_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate dense embedding: {e}")
            raise

    def get_dense_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.dense_model.get_text_embedding_batch(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate dense embeddings: {e}")
            raise

    def get_sparse_embedding(self, text: str) -> dict[int, float]:
        try:
            # BM25 expects a list of documents
            self.sparse_model.fit([text])
            sparse_embedding = self.sparse_model.encode_documents([text])
            # Convert scipy sparse array to dictionary format expected by Milvus
            row = sparse_embedding[0].tocoo()
            sparse_dict = {int(col): float(data) for col, data in zip(row.col, row.data)}
            return sparse_dict
        except Exception as e:
            logger.error(f"Failed to generate sparse embedding: {e}")
            raise

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict[int, float]]:
        """Generate sparse embeddings for multiple texts using BM25.

        Args:
            texts: List of input texts

        Returns:
            List of sparse embeddings
        """
        try:
            self.sparse_model.fit(texts)
            sparse_embeddings = self.sparse_model.encode_documents(texts)
            # Convert scipy sparse array to list of dictionaries
            result = []
            for i in range(sparse_embeddings.shape[0]):
                row = sparse_embeddings[i].tocoo()
                sparse_dict = {int(col): float(data) for col, data in zip(row.col, row.data)}
                result.append(sparse_dict)
            return result
        except Exception as e:
            logger.error(f"Failed to generate sparse embeddings: {e}")
            raise

    def get_query_sparse_embedding(self, query: str) -> dict[int, float]:
        """Generate sparse embedding for a query using BM25.

        Args:
            query: Query text

        Returns:
            Sparse embedding as dictionary {index: value}
        """
        try:
            # BM25 has separate method for queries
            sparse_embedding = self.sparse_model.encode_queries([query])
            # Convert scipy sparse array to dictionary format expected by Milvus
            row = sparse_embedding[0].tocoo()
            sparse_dict = {int(col): float(data) for col, data in zip(row.col, row.data)}
            return sparse_dict
        except Exception as e:
            logger.error(f"Failed to generate query sparse embedding: {e}")
            raise
