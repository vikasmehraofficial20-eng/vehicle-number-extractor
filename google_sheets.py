"""
Persists every detected vehicle number to a single master Google Sheet,
so audit data survives Render restarts (the local disk is ephemeral on
the free plan) and all audits accumulate in one place instead of being
scattered across separate per-job .xlsx files.

Setup required (see README notes / chat instructions):
  1. Create a Google Cloud service account with the Sheets API enabled.
  2. Share the target Google Sheet with the service account's email
     (Editor access).
  3. Set these two environment variables on Render:
       GOOGLE_SERVICE_ACCOUNT_JSON  -> full contents of the service
                                        account's downloaded JSON key
       GOOGLE_SHEET_ID              -> the sheet ID from its URL
"""
import os
import json
import threading
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HEADERS = ['Timestamp', 'Job ID', 'City', 'Garage/Location', 'Auditor Name', 'Date',
           'Vehicle Number', 'Confidence (%)', 'Times Detected', 'First Seen', 'Source']

_lock = threading.Lock()
_client = None
_worksheet = None


def _get_client():
    global _client
    if _client is None:
        raw = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
        if not raw:
            raise RuntimeError('GOOGLE_SERVICE_ACCOUNT_JSON env var not set')
        info = json.loads(raw)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        _client = gspread.authorize(creds)
    return _client


def _get_worksheet():
    global _worksheet
    if _worksheet is None:
        sheet_id = os.environ.get('GOOGLE_SHEET_ID')
        if not sheet_id:
            raise RuntimeError('GOOGLE_SHEET_ID env var not set')
        client = _get_client()
        spreadsheet = client.open_by_key(sheet_id)
        ws = spreadsheet.sheet1
        existing_header = ws.row_values(1)
        if existing_header != HEADERS:
            if not existing_header:
                ws.append_row(HEADERS)
            else:
                # Header exists but doesn't match — insert our header at the
                # top only if the sheet is otherwise empty, to avoid
                # clobbering data already in a sheet someone reused.
                if ws.row_count <= 1 and not any(existing_header):
                    ws.append_row(HEADERS)
        _worksheet = ws
    return _worksheet


def append_results(results, job_id, city, garage, auditor, audit_date, source_name):
    """Append one row per detected vehicle to the master Google Sheet.
    Safe to call from a background job thread. Any failure here is
    logged but never raised, so a Google Sheets outage can't break the
    local report generation the user is waiting on."""
    if not results:
        return
    with _lock:
        try:
            ws = _get_worksheet()
        except Exception as e:
            print(f"[GSHEET] Failed to connect to Google Sheet: {e}")
            return

        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        rows = []
        for r in results:
            rows.append([
                now,
                job_id,
                city,
                garage,
                auditor,
                audit_date,
                r['plate_number'],
                r['confidence'],
                r['frames_detected'],
                r['first_seen_seconds'],
                source_name,
            ])
        try:
            ws.append_rows(rows, value_input_option='USER_ENTERED')
            print(f"[GSHEET] Appended {len(rows)} row(s) for job {job_id}")
        except Exception as e:
            print(f"[GSHEET] Failed to append rows: {e}")
