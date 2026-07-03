"""
Uploads each audit's original photo/video files to a shared Google Drive
folder and returns a shareable "view" link for each one, so the master
Google Sheet can link directly to the actual uploaded file instead of
just showing its filename.

Setup required (in addition to the Google Sheets setup):
  1. Enable the Google Drive API on the same Google Cloud project used
     for Sheets.
  2. Create a Drive folder and share it with the service account's
     client_email (Editor access) — this makes files the service account
     uploads into that folder visible to you automatically.
  3. Set this environment variable on Render:
       GOOGLE_DRIVE_FOLDER_ID  -> the folder ID from its Drive URL
"""
import os
import json
import mimetypes
import threading

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

_lock = threading.Lock()
_service = None


def _get_service():
    global _service
    if _service is None:
        raw = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
        if not raw:
            raise RuntimeError('GOOGLE_SERVICE_ACCOUNT_JSON env var not set')
        info = json.loads(raw)
        creds = Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
        _service = build('drive', 'v3', credentials=creds)
    return _service


def upload_file(local_path, display_name=None):
    """Upload a local file to the shared Drive folder and return a
    shareable view link, or None on any failure. Never raises — a Drive
    outage should not block report generation or Sheet logging."""
    folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
    if not folder_id:
        print("[GDRIVE] GOOGLE_DRIVE_FOLDER_ID not set, skipping upload")
        return None
    if not os.path.exists(local_path):
        print(f"[GDRIVE] Local file missing, cannot upload: {local_path}")
        return None

    try:
        with _lock:
            service = _get_service()

        name = display_name or os.path.basename(local_path)
        mime_type, _ = mimetypes.guess_type(local_path)
        mime_type = mime_type or 'application/octet-stream'

        file_metadata = {'name': name, 'parents': [folder_id]}
        media = MediaFileUpload(local_path, mimetype=mime_type, resumable=False)
        uploaded = service.files().create(
            body=file_metadata, media_body=media, fields='id, webViewLink'
        ).execute()

        link = uploaded.get('webViewLink')
        print(f"[GDRIVE] Uploaded '{name}' -> {link}")
        return link
    except Exception as e:
        print(f"[GDRIVE] Upload failed for {local_path}: {e}")
        return None
