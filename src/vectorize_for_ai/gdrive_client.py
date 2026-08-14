import logging
import os
from typing import Dict, Iterator, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from vectorize_for_ai.config import gdrive_settings

logger = logging.getLogger(__name__)


class GDriveClient:
    def __init__(self):
        self.service = self._authenticate()

    def _authenticate(self) -> build:
        if not os.path.exists(gdrive_settings.drive_credentials_path):
            raise FileNotFoundError(f"Credentials not found: {self.credentials_path}")

        credentials = service_account.Credentials.from_service_account_file(
            gdrive_settings.drive_credentials_path, scopes=gdrive_settings.drive_scopes
        )
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    def list_files_incremental(
        self,
        drive_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        since_date: Optional[str] = None,
        date_field: str = "createdTime",  # or "modifiedTime"
        page_size: int = 100
    ) -> Iterator[Dict]:
        """
        Stream files one at a time. Only returns files created/modified after since_date.
        since_date: ISO 8601 string, e.g. '2024-01-15T10:30:00'
        """
        search_query = "trashed=false"

        if folder_id:
            search_query += f" and '{folder_id}' in parents"
        elif drive_id:
            search_query += f" and '{drive_id}' in parents"

        if since_date:
            # Google Drive API uses RFC 3339 format
            search_query += f" and {date_field} > '{since_date}'"
            logger.info(f"Filtering for files {date_field} > {since_date}")

        page_token = None

        while True:
            try:
                params = {
                    "q": search_query,
                    "pageSize": page_size,
                    "fields": "nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink, parents, driveId, md5Checksum)",
                    "supportsAllDrives": True,
                    "includeItemsFromAllDrives": True,
                    "orderBy": f"{date_field} asc",  # Oldest first
                }

                if drive_id:
                    params["driveId"] = drive_id
                    params["corpora"] = "drive"

                if page_token:
                    params["pageToken"] = page_token

                results = self.service.files().list(**params).execute()
                files = results.get("files", [])

                for file in files:
                    yield file

                page_token = results.get("nextPageToken")
                if not page_token:
                    break

            except HttpError as e:
                logger.error(f"Error listing files: {e}")
                raise

    def get_file_metadata(self, file_id: str) -> Dict:
        return self.service.files().get(
            fileId=file_id,
            supportsAllDrives=True,
            fields="id, name, mimeType, size, createdTime, modifiedTime, webViewLink, parents, driveId, md5Checksum, description"
        ).execute()

    def get_file_content_stream(self, file_id: str, mime_type: str):
        """
        Returns a file-like object for reading content streamingly.
        For Google Workspace files, exports to a readable format.
        For binary files, returns raw media stream.
        """
        if mime_type.startswith("application/vnd.google-apps."):
            # Export Google Workspace files to plain text or PDF
            export_mimes = {
                "application/vnd.google-apps.document": "text/plain",
                "application/vnd.google-apps.spreadsheet": "text/csv",
                "application/vnd.google-apps.presentation": "text/plain",
            }
            export_mime = export_mimes.get(mime_type, "application/pdf")

            request = self.service.files().export(fileId=file_id, mimeType=export_mime)
        else:
            request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)

        # Return the request object which supports streaming read
        return request
