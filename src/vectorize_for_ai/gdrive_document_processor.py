import io
from pathlib import Path
from typing import Callable, Dict, Optional

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from vectorize_for_ai.config import gdrive_settings
from vectorize_for_ai.gdrive_client import GDriveClient
from vectorize_for_ai.gdrive_state_manager import GDriveStateManager
from vectorize_for_ai.logger import get_logger

logger = get_logger(__name__)



class GDriveDocumentProcessor:
    def __init__(
        self,
        client: GDriveClient | None = None,
        state: GDriveStateManager | None = None,
        download_dir: Path = Path(gdrive_settings.drive_local_download_dir)
    ):
        self.client = client or GDriveClient()
        self.state = state or GDriveStateManager()
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)

    def process_one(
        self,
        file_id: str,
        handler: Optional[Callable[[bytes, Dict], None]] = None,
        download: bool = False
    ) -> Optional[Dict]:
        """
        Fetch and process a single document. Skips folders and non-exportable files gracefully.
        Returns metadata on success, None on skip/failure.
        """
        metadata = self.client.get_file_metadata(file_id)
        name = metadata["name"]
        mime_type = metadata["mimeType"]
        created_time = metadata["createdTime"]

        # SKIP FOLDERS
        if mime_type == self.client.folder_mime:
            logger.debug(f"Skipping folder: {name}")
            return None

        logger.info(f"Processing: {name} ({mime_type}) created {created_time}")

        # Get content stream
        try:
            request = self.client.get_file_content_stream(file_id, mime_type)
        except ValueError as e:
            logger.warning(f"Skipping {name}: {e}")
            return None

        # Stream download
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        try:
            done = False
            while not done:
                status, done = downloader.next_chunk()

            fh.seek(0)
            content = fh.read()

        except HttpError as e:
            # Gracefully handle non-exportable / restricted files
            if e.resp.status == 403 and "fileNotExportable" in str(e):
                logger.warning(f"Skipping {name}: file is not exportable (403 fileNotExportable)")
                return None
            if e.resp.status == 403:
                logger.warning(f"Skipping {name}: permission denied (403)")
                return None
            raise  # Re-raise unexpected errors

        # Call user handler
        if handler:
            handler(content, metadata)

        # Save to disk if requested
        if download:
            safe_name = self._sanitize_filename(name)
            if mime_type == "text/plain" and not safe_name.endswith(".txt"):
                safe_name += ".txt"
            elif mime_type == "text/csv" and not safe_name.endswith(".csv"):
                safe_name += ".csv"

            dest_path = self.download_dir / safe_name
            with open(dest_path, "wb") as f:
                f.write(content)
            logger.info(f"Saved to: {dest_path}")

        # Update state
        self.state.mark_processed(file_id)
        current_last = self.state.get_since_date()
        if not current_last or created_time > current_last:
            self.state.update_timestamp(created_time)

        return metadata

    def stream_new_documents(
        self,
        drive_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        since_date: Optional[str] = None,
        date_field: str = "createdTime",
        handler: Optional[Callable[[bytes, Dict], None]] = None,
        download: bool = False,
        force: bool = False
    ):
        """
        Generator that recursively yields one new document at a time.
        """
        since = self.state.get_since_date(since_date)
        logger.info(f"Streaming documents {date_field} > {since or 'BEGINNING'}")

        for file_meta in self.client.list_files_incremental(
            drive_id=drive_id,
            folder_id=folder_id,
            since_date=since,
            date_field=date_field
        ):
            file_id = file_meta["id"]

            if self.state.is_processed(file_id):
                continue

            result = self.process_one(
                file_id=file_id,
                handler=handler,
                download=download
            )

            if result:
                yield result
            # If result is None (folder or non-exportable), just continue to next file
