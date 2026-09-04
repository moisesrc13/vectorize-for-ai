import os
from typing import Dict, Iterator, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build

from vectorize_for_ai.config import gdrive_settings
from vectorize_for_ai.logger import get_logger

logger = get_logger(__name__)


class GDriveClient:
    def __init__(self):
        self.service = self._authenticate()
        self.folder_mime = "application/vnd.google-apps.folder"

        # Exportable Google Workspace files → preferred export format
        self.export_map = {
            "application/vnd.google-apps.document": "text/plain",
            "application/vnd.google-apps.spreadsheet": "text/csv",
            "application/vnd.google-apps.presentation": "application/pdf",
            "application/vnd.google-apps.drawing": "image/png",
        }

    def _authenticate(self) -> build:
        if not os.path.exists(gdrive_settings.drive_credentials_path):
            raise FileNotFoundError(f"Credentials not found: {self.credentials_path}")

        credentials = service_account.Credentials.from_service_account_file(
            gdrive_settings.drive_credentials_path, scopes=gdrive_settings.drive_scopes
        )
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    def _list_files_raw(
        self,
        q: str,
        drive_id: Optional[str] = None,
        page_size: int = 100
    ) -> Iterator[Dict]:
        """Low-level paginated file list."""
        page_token = None
        while True:
            params = {
                "q": q,
                "pageSize": page_size,
                "fields": "nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink, parents, driveId, md5Checksum)",
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
            }
            if drive_id:
                params["driveId"] = drive_id
                params["corpora"] = "drive"
            if page_token:
                params["pageToken"] = page_token

            results = self.service.files().list(**params).execute()
            yield from results.get("files", [])

            page_token = results.get("nextPageToken")
            if not page_token:
                break

    @staticmethod
    def _to_rfc3339(date_str: str) -> str:
        """Ensure a date string is RFC 3339 so Drive API comparisons work correctly.

        Drive stores timestamps as RFC 3339 datetimes.  Bare date strings like
        '2026-01-01' are silently unmatched by the API, returning 0 results.
        """
        if date_str and "T" not in date_str:
            return f"{date_str}T00:00:00Z"
        return date_str

    def list_files_incremental(
        self,
        drive_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        since_date: Optional[str] = None,
        date_field: str = "createdTime",
    ) -> Iterator[Dict]:
        """
        Recursively stream files from a drive or folder.
        Only returns files (not folders) matching the date filter.
        """
        since_date = self._to_rfc3339(since_date) if since_date else since_date

        # CASE 0: No drive_id or folder_id — search across all drives the service account can access
        if not drive_id and not folder_id:
            query = "trashed=false"
            if since_date:
                query += f" and {date_field} >= '{since_date}'"
            query += f" and mimeType != '{self.folder_mime}'"

            logger.info(f"Searching all drives for files {date_field} > {since_date or 'ALL TIME'}")
            page_token = None
            while True:
                params = {
                    "q": query,
                    "pageSize": 100,
                    "fields": "nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink, parents, driveId, md5Checksum)",
                    "supportsAllDrives": True,
                    "includeItemsFromAllDrives": True,
                    "corpora": "allDrives",
                }
                if page_token:
                    params["pageToken"] = page_token
                results = self.service.files().list(**params).execute()
                yield from results.get("files", [])
                page_token = results.get("nextPageToken")
                if not page_token:
                    break
            return

        # CASE 1: Entire shared drive — API can search across all folders at once
        if drive_id and not folder_id:
            query = "trashed=false"
            if since_date:
                query += f" and {date_field} >= '{since_date}'"
            # Exclude folders from results
            query += f" and mimeType != '{self.folder_mime}'"

            logger.info(f"Searching entire drive {drive_id} for files {date_field} > {since_date or 'ALL TIME'}")
            yield from self._list_files_raw(q=query, drive_id=drive_id)
            return

        # CASE 2: Specific folder — recursive BFS through subfolders
        def recurse(current_folder_id: str, path: str = ""):
            # Files in this folder matching date filter
            file_query = f"trashed=false and '{current_folder_id}' in parents and mimeType != '{self.folder_mime}'"
            if since_date:
                file_query += f" and {date_field} >= '{since_date}'"

            for file in self._list_files_raw(q=file_query, drive_id=drive_id):
                file["folder_path"] = path
                yield file

            # Find subfolders to recurse into (traverse ALL folders regardless of date)
            folder_query = f"trashed=false and '{current_folder_id}' in parents and mimeType = '{self.folder_mime}'"
            for subfolder in self._list_files_raw(q=folder_query, drive_id=drive_id):
                sub_name = subfolder["name"]
                new_path = f"{path}/{sub_name}" if path else sub_name
                yield from recurse(subfolder["id"], new_path)

        if folder_id:
            logger.info(f"Recursively scanning folder {folder_id} for files {date_field} > {since_date or 'ALL TIME'}")
            yield from recurse(folder_id)

    def get_file_metadata(self, file_id: str) -> Dict:
        return self.service.files().get(
            fileId=file_id,
            supportsAllDrives=True,
            fields="id, name, mimeType, size, createdTime, modifiedTime, webViewLink, parents, driveId, md5Checksum, description"
        ).execute()

    def get_file_content_stream(self, file_id: str, mime_type: str):
        """
        Returns a request object for downloading/exporting.
        Raises ValueError for folders or non-exportable files.
        """
        if mime_type == self.folder_mime:
            raise ValueError("Cannot download a folder")

        # Google Workspace file — try export
        if mime_type.startswith("application/vnd.google-apps."):
            export_mime = self.export_map.get(mime_type)
            if not export_mime:
                raise ValueError(f"No export format defined for {mime_type}")
            return self.service.files().export(fileId=file_id, mimeType=export_mime)

        # Regular binary file
        return self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
