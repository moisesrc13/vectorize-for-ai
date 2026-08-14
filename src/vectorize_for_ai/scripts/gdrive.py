import argparse
import logging
from pathlib import Path

from vectorize_for_ai.config import gdrive_settings
from vectorize_for_ai.gdrive_client import GDriveClient
from vectorize_for_ai.gdrive_document_processor import GDriveDocumentProcessor
from vectorize_for_ai.gdrive_state_manager import GDriveStateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Example handlers - replace with your own logic
def print_handler(content: bytes, metadata: dict):
    """Just prints info about the document."""
    print(f"\n--- Document: {metadata['name']} ---")
    print(f"Type: {metadata['mimeType']}")
    print(f"Size: {len(content)} bytes")
    print(f"Preview: {content[:200]}...")
    print("---")


def save_text_handler(content: bytes, metadata: dict):
    """Saves text content to a file for indexing."""
    output_dir = Path("./extracted_text")
    output_dir.mkdir(exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in metadata['name'])
    out_path = output_dir / f"{safe_name}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content.decode("utf-8", errors="replace"))

    print(f"Extracted text saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stream and process Google Drive documents one at a time"
    )
    parser.add_argument("--since", help="Only process files created after this date (ISO 8601: 2024-01-15T00:00:00)")
    args = parser.parse_args()

    client = GDriveClient()
    state = GDriveStateManager()
    processor = GDriveDocumentProcessor(client=client, state=state)

    since = args.since

    # Process documents ONE AT A TIME
    count = 0
    for metadata in processor.stream_new_documents(
        drive_id=gdrive_settings.drive_shared_id,
        folder_id=gdrive_settings.drive_folder_id,
        since_date=since,
        date_field="modifiedTime",
        handler=print_handler,  # <-- Replace with your custom handler
        download=gdrive_settings.drive_local_download_dir
    ):
        count += 1
        # You can access metadata here and do additional per-file work
        # The file has already been processed by your handler above

        # Optional: Stop after N files for testing
        # if count >= 5:
        #     print("Reached limit, stopping.")
        #     break

    print(f"\nDone! Processed {count} new document(s).")
    print(f"Last processed timestamp: {state.get_since_date()}")


if __name__ == "__main__":
    main()
