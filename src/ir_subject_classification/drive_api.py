"""Google Drive API access for folders too large for the mounted filesystem."""

from __future__ import annotations

import io
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

FOLDER_MIME = "application/vnd.google-apps.folder"
LOGGER = logging.getLogger(__name__)


def create_drive_service():
    try:
        import google.auth
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError("Google Drive API dependencies are required for fulltext_format=gdrive_api") from exc
    credentials, _ = google.auth.default()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def normalize_my_drive_path(path: str | Path) -> list[str]:
    value = str(path).replace("\\", "/").rstrip("/")
    markers = ("/content/drive/MyDrive/", "/content/drive/My Drive/")
    for marker in markers:
        if marker in value:
            return [part for part in value.split(marker, 1)[1].split("/") if part]
    raise ValueError(f"Google Drive path must be below MyDrive: {path}")


def resolve_folder_id(service, path: str | Path) -> str:
    parent = "root"
    for name in normalize_my_drive_path(path):
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        query = (
            f"'{parent}' in parents and trashed=false and "
            f"mimeType='{FOLDER_MIME}' and name='{escaped}'"
        )
        response = service.files().list(q=query, pageSize=100, fields="files(id,name)").execute()
        matches = response.get("files", [])
        if not matches:
            raise FileNotFoundError(f"Drive API could not find folder segment {name!r} below parent {parent}")
        if len(matches) > 1:
            raise RuntimeError(
                f"Drive contains {len(matches)} folders named {name!r} at the same level. "
                "Consolidate the duplicate folders before using path-based resolution."
            )
        parent = matches[0]["id"]
    return parent


def list_txt_files(service, folder_id: str, recursive: bool = True) -> pd.DataFrame:
    queue = [folder_id]
    rows: list[dict[str, object]] = []
    while queue:
        parent = queue.pop(0)
        token = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{parent}' in parents and trashed=false",
                    pageSize=1000,
                    pageToken=token,
                    fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)",
                )
                .execute()
            )
            for item in response.get("files", []):
                if item.get("mimeType") == FOLDER_MIME:
                    if recursive:
                        queue.append(item["id"])
                elif item.get("name", "").lower().endswith(".txt"):
                    rows.append(
                        {
                            "drive_file_id": item["id"],
                            "txt_filename": item["name"],
                            "file_id": Path(item["name"]).stem,
                            "size": item.get("size"),
                            "modified_time": item.get("modifiedTime"),
                        }
                    )
            token = response.get("nextPageToken")
            if not token:
                break
    return pd.DataFrame(rows)


def build_or_load_drive_index(
    folder_path: str | Path, cache_path: str | Path | None = None, refresh: bool = False
) -> tuple[pd.DataFrame, object]:
    cache = Path(cache_path) if cache_path else None
    service = create_drive_service()
    if cache and cache.is_file() and not refresh:
        return pd.read_csv(cache, dtype=str), service
    folder_id = resolve_folder_id(service, folder_path)
    index = list_txt_files(service, folder_id)
    if index.empty:
        raise RuntimeError(
            f"No TXT files were found through the Google Drive API under: {folder_path}"
        )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(cache, index=False)
    return index, service


def download_text(service, drive_file_id: str, retries: int = 3) -> str:
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise ImportError("google-api-python-client is required") from exc
    for attempt in range(1, retries + 1):
        try:
            stream = io.BytesIO()
            downloader = MediaIoBaseDownload(
                stream, service.files().get_media(fileId=drive_file_id), chunksize=1024 * 1024
            )
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return stream.getvalue().decode("utf-8", errors="replace")
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)
    return ""


def read_selected_fulltext(
    mapped: pd.DataFrame,
    handles: list[str] | pd.Series,
    max_chars: int = 100_000,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    wanted = set(map(str, handles))
    selected = mapped[mapped["handle"].astype(str).isin(wanted)]
    service = create_drive_service()
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    texts: dict[str, list[str]] = defaultdict(list)
    for row in selected.itertuples(index=False):
        cached_path = cache / f"{row.drive_file_id}.txt" if cache else None
        try:
            if cached_path and cached_path.is_file():
                text = cached_path.read_text(encoding="utf-8", errors="replace")
            else:
                text = download_text(service, str(row.drive_file_id))
                if cached_path:
                    temporary = cached_path.with_suffix(".tmp")
                    temporary.write_text(text, encoding="utf-8")
                    os.replace(temporary, cached_path)
        except Exception as exc:
            LOGGER.warning(
                "Could not download Drive file %s (%s): %s",
                row.drive_file_id,
                getattr(row, "txt_filename", "unknown"),
                exc,
            )
            continue
        if text.strip():
            texts[str(row.handle)].append(text)
    rows = []
    for handle, parts in texts.items():
        complete_text = "\n\n".join(parts)
        rows.append(
            {
                "handle": handle,
                "fulltext": complete_text[:max_chars],
                "fulltext_source_characters": len(complete_text),
                "fulltext_truncated": len(complete_text) > max_chars,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "handle",
            "fulltext",
            "fulltext_source_characters",
            "fulltext_truncated",
        ],
    )
