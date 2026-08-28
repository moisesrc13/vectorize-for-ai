import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import FormatToExtensions, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from llama_index.core.schema import TextNode
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader

from vectorize_for_ai.database_factory import DatabaseFactory
from vectorize_for_ai.embeddings import EmbeddingHandler
from vectorize_for_ai.logger import get_logger

logger = get_logger(__name__)



def remove_pdf_decorations(pdf_bytes: bytes) -> bytes | None:
    """Cleans PDF decorations from input bytes and returns the resulting PDF as bytes."""
    try:
        import ghostscript
    except (ImportError, RuntimeError) as e:
        print(
            f"Warning: Ghostscript is not available ({e}). Falling back to default backend."
        )
        return None
    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_in,
        tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_out,
    ):
        try:
            temp_in.write(pdf_bytes)
            temp_in.close()
            temp_out.close()

            args = [
                "ps2pdf",
                "-dNOPAUSE",
                "-dBATCH",
                "-dSAFER",
                "-sDEVICE=pdfwrite",
                "-dPreserveAnnots=false",
                f"-sOutputFile={temp_out.name}",
                temp_in.name,
            ]

            ghostscript.Ghostscript(*args)

            if (
                not pathlib.Path(temp_out.name).exists()
                or pathlib.Path(temp_out.name).stat().st_size == 0
            ):
                return None

            return pathlib.Path(temp_out.name).read_bytes()

        finally:
            for path in [temp_in.name, temp_out.name]:
                if pathlib.Path(path).exists():
                    with contextlib.suppress(OSError):
                        pathlib.Path(path).unlink()


class TiktokenTokenizer(BaseTokenizer):
    """Tiktoken-based tokenizer for use with HybridChunker.

    Uses OpenAI's tiktoken library (cl100k_base encoding, same as GPT-4/gpt-4o)
    to count tokens, giving accurate chunk sizes when content will be sent to
    OpenAI-compatible models.
    """

    encoding_name: str = "cl100k_base"
    max_tokens: int = 512

    def count_tokens(self, text: str) -> int:
        enc = tiktoken.get_encoding(self.encoding_name)
        return len(enc.encode(text))

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding(self.encoding_name)


class DocumentIngestionPipeline:

    def __init__(self, chunk_max_tokens: int = 512) -> None:
        self.cos_client = COSClient()
        self.db_handler = DatabaseFactory.create_handler()
        self.embedding_handler = EmbeddingHandler()
        self.semantic_splitter = None
        self.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device="cpu",
        )

        # Initialize DoclingNodeParser with tiktoken-based chunker for
        # accurate token counting (cl100k_base / GPT-4 encoding)
        tokenizer = TiktokenTokenizer(max_tokens=chunk_max_tokens)
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        self.node_parser = DoclingNodeParser(
            chunker=chunker,
        )
        logger.info(
            "Initialized DoclingNodeParser with TiktokenTokenizer (cl100k_base, max_tokens=%d)",
            chunk_max_tokens,
        )

    def _create_pipeline_options(self) -> PdfPipelineOptions:
        """Create pipeline options with accelerator configuration."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = self.accelerator_options
        return pipeline_options

    def _create_metadata_dict(
        self, filename: str, snow_system: SNOWSystem | None
    ) -> dict[str, Any]:
        """Create metadata dictionary for a node.

        Args:
            filename: The filename to use in metadata
            snow_system: Optional SNOWSystem object with metadata

        Returns:
            Dictionary containing metadata fields
        """
        if snow_system:
            return {
                "attachment_name": filename,
                "name": snow_system.name,
                "ai_services": snow_system.ai_services or "",
                "agentic_solution_type": snow_system.agentic_solution_type or "",
                "ai_system_id": snow_system.ai_system_id,
                "sys_created_on": snow_system.sys_created_on,
                "short_description": snow_system.short_description or "",
                "expected_benefit": snow_system.expected_benefit or "",
            }
        else:
            return {
                "attachment_name": filename,
                "name": os.path.splitext(os.path.basename(Path(filename).name))[0],
                "ai_services": "",
                "agentic_solution_type": "",
                "ai_system_id": "",
                "sys_created_on": "",
                "short_description": "",
                "expected_benefit": "",
            }

    def _format_snow_metadata_text(self, snow_system: SNOWSystem) -> str:
        """Format SNOW metadata as pipe-separated text.

        Args:
            snow_system: SNOWSystem object with metadata

        Returns:
            Pipe-separated string of name, description, and benefit
        """
        return f"{snow_system.name}|{snow_system.short_description}|{snow_system.expected_benefit}"

    def _create_embedding_text(self, node: TextNode) -> str:
        """Create text for embedding generation from a node.

        Args:
            node: TextNode to create embedding text from

        Returns:
            Formatted text combining content and metadata
        """
        return (
            f"{node.get_content()}|"
            f"{node.metadata.get('name', '')}|"
            f"{node.metadata.get('short_description', '')}|"
            f"{node.metadata.get('expected_benefit', '')}"
        )

    def _create_metadata_node(self, filename: str, snow_system: SNOWSystem) -> TextNode:
        """Create a metadata-only node from SNOW system data.

        Args:
            filename: The filename for metadata
            snow_system: SNOWSystem object with metadata

        Returns:
            TextNode with metadata and embedding
        """
        metadata_text = self._format_snow_metadata_text(snow_system)
        metadata_dict = self._create_metadata_dict(filename, snow_system)

        node = TextNode(text=metadata_text, metadata=metadata_dict)

        # Generate and attach dense embedding
        dense_embedding = self.embedding_handler.get_dense_embeddings([metadata_text])[0]
        node.embedding = dense_embedding

        return node

    def _delete_nodes_by_system_number(self, ai_system_number: str) -> int:
        """delete nodes by ai system id number"""
        return self.db_handler.delete_nodes_by_system_number(ai_system_number)

    def _insert_nodes(self, nodes: list[Any], document_name: str) -> list[str]:
        """Insert nodes into database and log the operation.

        Args:
            nodes: List of nodes to insert
            document_name: Name of the document for logging

        Returns:
            List of node IDs
        """
        if not nodes:
            logger.info("empty nodes")
            return []
        node_ids = self.db_handler.add_nodes(nodes)
        logger.info(f"Inserted {len(node_ids)} nodes for document: {document_name}")
        return node_ids

    def _get_converter_pipeline(self, file_extension: str) -> DocumentConverter | None:
        try:
            if file_extension in FormatToExtensions.get(InputFormat.IMAGE):
                file_extension = InputFormat.IMAGE.value
            input_format = InputFormat(file_extension)
        except Exception as err:
            logger.error("input format not support for %s - %s", file_extension, err)
            return None

        converter = None
        try:
            match input_format:
                case InputFormat.PDF:
                    pipeline_options = self._create_pipeline_options()
                    pipeline_options.do_ocr = True
                    pipeline_options.do_table_structure = True
                    pipeline_options.table_structure_options.do_cell_matching = True
                    converter = DocumentConverter(
                        format_options={
                            input_format: PdfFormatOption(
                                pipeline_options=pipeline_options,
                            )
                        }
                    )
                case InputFormat.PPTX:
                    pipeline_options = self._create_pipeline_options()
                    converter = DocumentConverter(
                        format_options={
                            input_format: PowerpointFormatOption(
                                pipeline_options=pipeline_options,
                            )
                        }
                    )
                case InputFormat.DOCX:
                    pipeline_options = self._create_pipeline_options()
                    converter = DocumentConverter(
                        format_options={
                            input_format: WordFormatOption(
                                pipeline_options=pipeline_options,
                            )
                        }
                    )
                case _:
                    pipeline_options = self._create_pipeline_options()
                    pipeline_options.do_picture_classification = True
                    pipeline_options.generate_parsed_pages = True
                    converter = DocumentConverter(
                        format_options={
                            input_format: ImageFormatOption(
                                pipeline_options=pipeline_options,
                            )
                        }
                    )
            return converter
        except Exception as err:
            logger.error("error getting converter for %s - %s", file_extension, err)
            return None

    def _process_document(
        self, filename: str, content: bytes, snow_system: SNOWSystem | None = None
    ) -> list[Any]:
        # Create temporary file for document
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # For PDFs, attempt to strip decorations before parsing
        if suffix.lower() == ".pdf":
            cleaned = remove_pdf_decorations(content)
            if cleaned:
                logger.info("PDF decorations removed for %s", filename)
                Path(tmp_path).write_bytes(cleaned)
            else:
                logger.debug("PDF decoration removal skipped for %s (ghostscript unavailable or returned empty)", filename)

        try:
            # Parse document with DoclingReader
            converter = self._get_converter_pipeline(suffix.replace(".", ""))
            if converter is None:
                logger.warning(f"No converter available for file type: {suffix}")
                return []

            reader = DoclingReader(converter=converter, export_type=DoclingReader.ExportType.JSON)
            documents = reader.load_data(file_path=tmp_path)

            if not documents:
                logger.warning(f"No documents loaded from {filename}")
                return []

            # Add metadata to documents
            metadata_dict = self._create_metadata_dict(filename, snow_system)
            for doc in documents:
                doc.metadata.update(metadata_dict)

            logger.info("Loaded %s documents from %s", len(documents), filename)

            # Parse documents into nodes using DoclingNodeParser
            doc_nodes = self.node_parser.get_nodes_from_documents(documents)
            nodes = self.embedding_handler.get_semantic_nodes(doc_nodes)
            logger.info("Created %s nodes from documents", len(nodes))

            if (
                not nodes
                and snow_system
                and snow_system.name
                and snow_system.short_description
                and snow_system.expected_benefit
            ):
                logger.info("No nodes created from document content, using SNOW metadata fallback")
                nodes = [self._create_metadata_node(filename, snow_system)]
            # Generate dense embeddings for each node
            # Note: Sparse embeddings are generated automatically by MilvusVectorStore
            # when enable_sparse=True (uses BM25 by default)
            logger.info("Generating dense embeddings for %s nodes", len(nodes))
            texts = [self._create_embedding_text(node) for node in nodes]
            dense_embeddings = self.embedding_handler.get_dense_embeddings(texts)

            # Attach dense embeddings to nodes
            for node, dense_emb in zip(nodes, dense_embeddings):
                node.embedding = dense_emb

            logger.info("Processed document %s: %s nodes with embeddings", filename, len(nodes))
            return nodes

        except Exception as e:
            logger.error("Failed to process document %s: %s", filename, e)
            raise
        finally:
            # Clean up temporary file
            try:
                Path(tmp_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")

    def ingest_cos_document(self, key: str) -> bool:
        try:
            logger.info(f"Starting ingestion for document: {key}")

            # Download document from COS
            content = self.cos_client.download_document(key)

            # Process document with DoclingReader and DoclingNodeParser
            nodes = self._process_document(key, content)

            if not nodes:
                logger.warning(f"No nodes extracted from document: {key}")
                return False

            # Insert nodes into database using LlamaIndex vector store
            self._insert_nodes(nodes, key)

            logger.info(f"Successfully ingested document: {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to ingest document {key}: {e}")
            return False

    def ingest_document_from_content(
        self, content: bytes | None, filename: str, snow_system: SNOWSystem | None = None
    ) -> bool:
        """Ingest a document from file content instead of COS.

        Args:
            content: The file content as bytes, or None if no attachment exists
            filename: The original filename (used for metadata and file type detection)
            snow_system: Optional SNOWSystem object containing ServiceNow metadata

        Returns:
            bool: True if ingestion was successful, False otherwise
        """
        try:
            logger.info(f"Starting ingestion for uploaded file: {filename}")

            # If no content but we have SNOW system metadata, create metadata-only node
            if not content and snow_system:
                logger.info(
                    f"No attachment content for {filename}, creating metadata-only node from SNOW system data"
                )

                if (
                    not snow_system.name
                    or not snow_system.short_description
                    or not snow_system.expected_benefit
                ):
                    logger.warning(f"Insufficient SNOW metadata to create node for {filename}")
                    return False

                nodes = [self._create_metadata_node(filename, snow_system)]
                logger.info(f"Created metadata-only node for system: {snow_system.name}")
            elif not content:
                logger.warning(f"No content and no SNOW metadata for {filename}")
                return False
            else:
                # Process document with DoclingReader and DoclingNodeParser
                nodes = self._process_document(filename, content, snow_system)

            if not nodes:
                logger.warning(f"No nodes extracted from uploaded file: {filename}")
                return False

            # Insert nodes into database using LlamaIndex vector store
            self._insert_nodes(nodes, filename)

            logger.info(f"Successfully ingested uploaded file: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to ingest uploaded file {filename}: {e}")
            return False

    def ingest_documents_from_cos(
        self, task_id: str, start_date: datetime | None = None
    ) -> dict[str, Any]:
        logger.info("Starting document ingestion pipeline")

        stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "failed_documents": [],
        }

        update_task_status(task_id, "processing", 0, "Task started", 0)
        for key, last_modified in self.cos_client.list_documents(start_date):
            stats["total"] += 1
            logger.info(
                f"Processing document {stats['total']}: {key} "
                f"(modified: {last_modified.isoformat()})"
            )

            # Ingest document (continue on failure)
            update_task_status(
                task_id,
                f"processing {key}",
                stats["total"],
                f"Processing document {key}",
                stats["total"],
            )
            success = self.ingest_cos_document(key)

            if success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
                stats["failed_documents"].append(key)

        logger.info(
            f"Ingestion complete. Total: {stats['total']}, "
            f"Successful: {stats['successful']}, Failed: {stats['failed']}"
        )
        update_task_status(
            task_id, TaskStatusEnum.completed, 100, "Task completed successfully", stats["total"]
        )
        return stats

    def close(self) -> None:
        self.db_handler.close()
        logger.info("Closed all connections")
