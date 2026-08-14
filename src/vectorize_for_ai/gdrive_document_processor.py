import io
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from googleapiclient.http import MediaIoBaseDownload

from vectorize_for_ai.config import gdrive_settings
from vectorize_for_ai.gdrive_client import GDriveClient
from vectorize_for_ai.gdrive_state_manager import GDriveStateManager

logger = logging.getLogger(__name__)


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

    def process_one(
        self,
        file_id: str,
        handler: Optional[Callable[[bytes, Dict], None]] = None,
        download: bool = False
    ) -> Dict:
        """
        Fetch and process a single document.

        Args:
            file_id: Google Drive file ID
            handler: Callback function(bytes, metadata) to process content
            download: If True, saves to disk; if False, just streams to handler

        Returns:
            File metadata dict
        """
        metadata = self.client.get_file_metadata(file_id)
        name = metadata["name"]
        mime_type = metadata["mimeType"]
        created_time = metadata["createdTime"]

        logger.info(f"Processing: {name} (created: {created_time})")

        # Get content stream
        request = self.client.get_file_content_stream(file_id, mime_type)

        # Stream download into memory (chunked)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        fh.seek(0)
        content = fh.read()

        # Call handler if provided
        if handler:
            handler(content, metadata)

        # Save to disk if requested
        if download:
            safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
            if not Path(safe_name).suffix:
                # Add extension based on mime type if missing
                ext_map = {
                    "text/plain": ".txt",
                    "text/csv": ".csv",
                    "application/pdf": ".pdf",
                }
                safe_name += ext_map.get(mime_type, "")

            dest_path = self.download_dir / safe_name
            with open(dest_path, "wb") as f:
                f.write(content)
            logger.info(f"Saved to: {dest_path}")

        # Update state tracking
        self.state.mark_processed(file_id)

        # Update timestamp if this is newer
        current_last = self.state.get_since_date()
        if not current_last or created_time > current_last:
            self.state.update_timestamp(created_time)

        return metadata

    def stream_new_documents(
        self,
        drive_id: str | None = None,
        folder_id: str | None = None,
        since_date: str | None = None,
        date_field: str = "createdTime",
        handler: Optional[Callable[[bytes, Dict], None]] = None,
        download: bool = False
    ):
        """
        Generator that yields one new document at a time.
        Only fetches documents created after since_date (or last run).
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

            # Skip if already processed (safety check)
            if self.state.is_processed(file_id):
                logger.debug(f"Skipping already processed: {file_meta['name']}")
                continue

            try:
                result = self.process_one(
                    file_id=file_id,
                    handler=handler,
                    download=download
                )
                yield result

            except Exception as e:
                logger.error(f"Failed to process {file_meta['name']}: {e}")
                # Continue to next file - don't let one failure stop the stream
                continue
