from datetime import datetime
from typing import Any, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    mode: Literal["loader", "api", "loader_and_api"] = Field(
        default="loader_and_api", description="Application mode (loader, api, loader_and_api)"
    )
    api_key: SecretStr = Field(..., description="API key for authentication")
    log_level: str = Field(default="INFO", description="Logging level")
    # loader config
    loader_start_date: str = Field(
        description="loader start date in ISO format (YYYY-MM-DD)",
        default=datetime.now().strftime("%Y-%m-%d"),
    )

    # Database Configuration
    database_type: Literal["opensearch"] = Field(
        default="opensearch",
        description="Database type to use for ingestion (opensearch)",
    )

    # OpenSearch Configuration
    opensearch_host: str = Field(default="localhost", description="OpenSearch host")
    opensearch_port: int = Field(default=9200, description="OpenSearch port")
    opensearch_user: str = Field(default="admin", description="OpenSearch username")
    opensearch_password: str = Field(default="", description="OpenSearch password")
    opensearch_index_name: str = Field(default="documents", description="OpenSearch index name")
    opensearch_use_ssl: bool = Field(default=False, description="Use SSL for OpenSearch connection")
    opensearch_verify_certs: bool = Field(default=False, description="Verify SSL certificates")
    opensearch_embedding_field: str = Field(
        default="embedding", description="OpenSearch field name for embeddings"
    )
    opensearch_text_field: str = Field(
        default="content", description="OpenSearch field name for text content"
    )
    embedding_dimensions: int = Field(default=384, description="Embedding vector dimensions")

    @property
    def opensearch_index_config(self) -> dict[str, Any]:
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "analysis": {"analyzer": {"default": {"type": "standard"}}},
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dimensions,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "faiss",
                            "parameters": {"ef_construction": 256, "m": 48},
                        },
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "term_vector": "with_positions_offsets",
                    },
                }
            },
        }

    @property
    def opensearch_uri(self) -> str:
        if self.opensearch_use_ssl:
            return f"https://{self.opensearch_host}:{self.opensearch_port}"
        return f"http://{self.opensearch_host}:{self.opensearch_port}"

    # Embedding Model Configuration
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace embedding model name",
    )

    # FastAPI Configuration
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8010, description="FastAPI port")
    api_title: str = Field(default="Hybrid Search API", description="FastAPI application title")
    api_version: str = Field(default="1.0.0", description="API version")

    # Processing Configuration
    chunk_max_tokens: int = Field(default=512, description="chunk max tokens for processing")

    # mcp
    mcp_host: str = Field(default="0.0.0.0", description="mcp server host")
    mcp_port: int = Field(default=8443, description="mcp server port")



class GDriveSettings(BaseSettings):
    drive_credentials_path: str = Field(default="credentials.json", description="drive credentials")
    drive_shared_id: str = Field(default="", description="shared drive id")
    drive_folder_id: str = Field(default="", description="shared drive folder id")
    drive_local_download_dir: str = Field(default="./downloads", description="tmp local dir for downloads")
    drive_scopes: list[str] = Field(
        default=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.metadata.readonly",
        ],
        description="drive scopes"
    )
    drive_state_file: str = Field(default=".gdrive_state.json", description="state drive")


# Global settings instance
settings = Settings()
gdrive_settings = GDriveSettings()
