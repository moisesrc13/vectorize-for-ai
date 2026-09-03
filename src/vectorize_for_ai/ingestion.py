import contextlib
import json
import pathlib
import tempfile
from pathlib import Path
from typing import Any

import tiktoken
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
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
from llama_index.core.schema import BaseNode, TextNode
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
        logger.warning(
            "Ghostscript is not available (%s). Falling back to default backend.", e
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

    def _create_embedding_text(self, node: TextNode) -> str:
        """Create text for embedding generation from a node.

        Args:
            node: TextNode to create embedding text from

        Returns:
            Formatted text combining content and metadata
        """
        return (
            f"{node.get_content()}|"
            f"{node.metadata.get('name', '')}"
        )


    # ---------------------------------------------------------------------------
    # Document ingestion (parsing → embedding → storage)
    # ---------------------------------------------------------------------------

    def ingest_document(
        self,
        content: bytes,
        metadata: dict,
        embedding_handler: EmbeddingHandler,
    ) -> int:
        """Parse *content*, generate dense + sparse embeddings, and upsert into the
        vector store.

        Args:
            content: Raw document bytes.
            metadata: Google Drive file metadata dict.
            embedding_handler: Handler used to generate embeddings.

        Returns:
            Number of nodes indexed.
        """
        file_name: str = metadata.get("name", "unknown")
        mime_type: str = metadata.get("mimeType", "")

        _MIME_TO_EXT: dict[str, str] = {
            "text/plain": ".txt",
            "text/csv": ".csv",
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "image/png": ".png",
            "image/jpeg": ".jpg",
        }
        suffix = _MIME_TO_EXT.get(mime_type, Path(file_name).suffix or ".bin")
        ext = suffix.lstrip(".")

        nodes: list[BaseNode] = []

        # Determine if the format is supported by Docling
        supported = False
        for fmt in InputFormat:
            exts = FormatToExtensions.get(fmt) or []
            if ext in exts or fmt.value == ext:
                supported = True
                break

        if supported:
            converter = self._get_converter_pipeline(ext)
            if converter is not None:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
                    documents = reader.load_data(file_path=tmp_path)
                    if documents:
                        for doc in documents:
                            doc.metadata.update(
                                {
                                    "file_name": file_name,
                                    "gdrive_id": metadata.get("id", ""),
                                    "created_time": metadata.get("createdTime", ""),
                                    "mime_type": mime_type,
                                }
                            )
                        nodes = self.node_parser.get_nodes_from_documents(documents)
                except Exception as parse_err:
                    logger.warning(
                        "Docling parsing failed for %s, falling back to plain-text node: %s",
                        file_name,
                        parse_err,
                    )
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

        # Fallback: single TextNode from raw text content
        if not nodes:
            try:
                text = content.decode("utf-8", errors="replace")
            except Exception:
                text = repr(content[:500])
            nodes = [
                TextNode(
                    text=text,
                    metadata={
                        "file_name": file_name,
                        "gdrive_id": metadata.get("id", ""),
                        "created_time": metadata.get("createdTime", ""),
                        "mime_type": mime_type,
                    },
                )
            ]

        if not nodes:
            logger.warning("No nodes extracted from %s", file_name)
            return 0

        # Remove 'origin' metadata to avoid OpenSearch numeric-overflow issues
        for node in nodes:
            node.metadata.pop("origin", None)

        # Dense embeddings
        texts = [node.get_content() for node in nodes]
        dense_embeddings = embedding_handler.get_dense_embeddings(texts)
        for node, dense_emb in zip(nodes, dense_embeddings):
            node.embedding = dense_emb

        # Sparse embeddings (BM25) stored as JSON string for hybrid search
        sparse_embeddings = embedding_handler.get_sparse_embeddings(texts)
        for node, sparse_emb in zip(nodes, sparse_embeddings):
            node.metadata["sparse_embedding"] = json.dumps(sparse_emb)

        node_ids = self._insert_nodes(nodes, file_name)
        return len(node_ids)

    def _delete_nodes_by_system_number(self, ai_system_number: str) -> int:
        """delete nodes by ai system id number"""
        return self.db_handler.delete_nodes_by_system_number(ai_system_number)

    async def delete_nodes_by_filename(self, file_name: str) -> int:
        """Delete all indexed nodes whose metadata.file_name matches *file_name*."""
        return await self.db_handler.delete_nodes_by_filename(file_name)

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
            if file_extension in (FormatToExtensions.get(InputFormat.IMAGE) or []):
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
                    pipeline_options.do_ocr = False
                    pipeline_options.do_table_structure = False
                    converter = DocumentConverter(
                        format_options={
                            input_format: PdfFormatOption(
                                pipeline_options=pipeline_options,
                                backend=PyPdfiumDocumentBackend,
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

    def close(self) -> None:
        self.db_handler.close()
        logger.info("Closed all connections")
