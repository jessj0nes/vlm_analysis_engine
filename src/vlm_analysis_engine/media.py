import errno
import glob
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from typing import Tuple, Optional, Union

import yt_dlp
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource

MODULE_ROOT = Path.cwd().parent.resolve()
logger = logging.getLogger(__name__)


def normalize_download_url(url: str) -> str:
    """
    Fix common malformed URLs so yt-dlp / gallery-dl extractors recognize them.

    X/Twitter permalinks use /status/<id>; some feeds use /statuses/<id>, which
    breaks extractors and yields "Unsupported URL" / generic extractor failures.
    """
    u = (url or "").strip()
    if not u or "/statuses/" not in u:
        return u
    low = u.lower()
    if "x.com" in low or "twitter.com" in low:
        return u.replace("/statuses/", "/status/")
    return u


def url_to_media_save_name(url: str, role: str = "media") -> str:
    """
    Stable filesystem-safe directory name derived from the download URL (SHA-256, first 16 hex chars).
    role distinguishes post-page vs direct media downloads when the same URL could appear in both paths.
    """
    raw = f"{role}\n{url.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16] + ("_post" if role == "post" else "")


def find_existing_downloaded_media(media_dir: str, save_name: str) -> Optional[str]:
    """
    If media was already downloaded for this save_name, return the file path; else None.
    Matches yt-dlp (save_name.mp4 under save_name/) and gallery-dl (save_name.* in save_name/).
    """
    base = os.path.join(media_dir, save_name)
    if not os.path.isdir(base):
        return None
    mp4 = os.path.join(base, f"{save_name}.mp4")
    if os.path.isfile(mp4) and os.path.getsize(mp4) > 0:
        return mp4
    matches = sorted(glob.glob(os.path.join(base, f"{save_name}*")))
    files = [f for f in matches if os.path.isfile(f) and os.path.getsize(f) > 0]
    return files[0] if files else None


def _download_video_with_ytdlp(url: str, media_dir: str, save_name: str, sep_audio: bool = False) -> Tuple[
    Optional[str], Optional[str]]:
    """
    Internal helper to attempt downloading media as a video using yt-dlp.
    Returns (path_to_file, error_message).
    """
    try:
        # Create a unique directory for the temporary download and merge
        save_path_template = f"{media_dir}/{save_name}/{save_name}.mp4"
        os.makedirs(os.path.dirname(save_path_template), exist_ok=True)

        final_path = os.path.join(media_dir, save_name, f"{save_name}.mp4")

        # Configure yt-dlp options
        if not sep_audio:
            ydl_opts = {
                'outtmpl': save_path_template,  # output file name template
                'format': 'bestvideo+bestaudio/best',  # best available quality
                'merge_output_format': 'mp4',
                'quiet': True,  # Suppress progress output for cleaner logs
                'no_warnings': True,
                'noprogress': True,
                'noplaylist': True,  # Important for non-playlist URLs
            }
        else:
            ydl_opts = {
                'outtmpl': save_path_template,
                'format': 'bestvideo+bestaudio/best',
                'keepvideo': True,
                'audio_multistreams': True,
                'merge_output_format': None,
                'quiet': True,
                'no_warnings': True,
                'noprogress': True,
            }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if os.path.exists(final_path):
            return final_path, None
        else:
            return None, "yt-dlp completed without creating expected MP4 file (likely non-video content)."

    except Exception as e:
        if os.path.isfile(final_path) and os.path.getsize(final_path) > 0:
            return final_path, None
        epipe = isinstance(e, BrokenPipeError) or (
            isinstance(e, OSError) and e.errno == errno.EPIPE
        )
        if epipe and not sep_audio:
            alt_tmpl = os.path.join(media_dir, save_name, f"{save_name}.%(ext)s")
            alt_opts = {
                'outtmpl': alt_tmpl,
                'format': 'best',
                'quiet': True,
                'no_warnings': True,
                'noprogress': True,
                'noplaylist': True,
            }
            try:
                with yt_dlp.YoutubeDL(alt_opts) as ydl:
                    ydl.download([url])
                cached = find_existing_downloaded_media(media_dir, save_name)
                if cached:
                    return cached, None
            except Exception as e2:
                return None, str(e2)
            return None, str(e)
        return None, str(e)


def _resolve_tool_path(name: str) -> str:
    """Locate a CLI tool, checking the active Python env's bin dir first."""
    env_bin = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.isfile(env_bin):
        return env_bin
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"Cannot locate {name!r}")


def _gallery_dl_argv0() -> list[str]:
    """CLI entry as argv prefix: resolved binary, or ``python -m gallery_dl``."""
    try:
        return [_resolve_tool_path("gallery-dl")]
    except FileNotFoundError:
        return [sys.executable, "-m", "gallery_dl"]


def _download_image_gallery_dl(url: str, media_dir: str, save_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Download an image/gallery via the gallery-dl CLI.
    Resolves the binary from the current Python environment so it works
    regardless of shell PATH configuration.
    """
    output_dir = os.path.join(media_dir, save_name)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        *_gallery_dl_argv0(),
        "-q",
        "--dest", output_dir,
        "--filename", save_name + ".{extension}",
        url,
    ]

    try:
        logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return None, f"gallery-dl exit {result.returncode}: {stderr}" if stderr else f"gallery-dl exit {result.returncode}"

        files = glob.glob(os.path.join(output_dir, "**", f"{save_name}*"), recursive=True)
        files = [f for f in files if os.path.isfile(f) and os.path.getsize(f) > 0]
        return (files[0], None) if files else (None, "gallery-dl completed but no files found.")

    except subprocess.TimeoutExpired:
        return None, "gallery-dl timed out after 120s"
    except Exception as e:
        return None, f"gallery-dl error: {e}"


def download_media(url: str, media_dir: str, save_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempts to download media, first as a video (yt-dlp), then as an image/gallery (gallery-dl).
    Returns (path_to_downloaded_file, error_message)

    Callers should pass the same URL they used to build ``save_name`` (see ``normalize_download_url``).
    """
    url = normalize_download_url(url.strip())
    cached = find_existing_downloaded_media(media_dir, save_name)
    if cached:
        logger.info("Reusing existing download for %s: %s", save_name, cached)
        return cached, None

    logger.info("Processing ID %s: Trying video download...", save_name)

    # 1. Try Video Download
    video_path, video_error = _download_video_with_ytdlp(url, media_dir, save_name)

    if video_path:
        logger.info("Video download successful.")
        return video_path, None

    # 2. Try Image/Gallery Download if video failed
    logger.info("Video download failed (%s). Trying image download with gallery-dl...", video_error)
    image_path, image_error = _download_image_gallery_dl(url, media_dir, save_name)

    if image_path:
        logger.info("Image download successful.")
        return image_path, None

    # 3. Both failed
    return None, f"Both video (Error: {video_error}) and image (Error: {image_error}) downloads failed."


def col_index_to_letter(col_index):
    """
    Google sheets column index to letter helper,
    """
    letter = ''
    while col_index >= 0:
        letter = chr(65 + col_index % 26) + letter
        col_index = col_index // 26 - 1
    return letter


def col_letter_to_index(letter: str) -> int:
    """
        Google sheets column letter to index helper,
    """
    index = 0
    for char in letter.upper():
        index = index * 26 + (ord(char) - ord('A') + 1)
    return index - 1


def _build_google_api_services(creds_path: str) -> dict[str, Resource]:
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/script.projects",
    ]
    token_path = os.path.join(creds_path, "token.json")
    secrets = os.path.join(creds_path, "client_secrets.json")
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(secrets, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    sheets_service = build("sheets", "v4", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)
    scripts_service = build("script", "v1", credentials=creds)

    return {"sheets": sheets_service, "drive": drive_service, "scripts": scripts_service}


def setup_api_services_for_credentials_dir(credentials_dir: Union[str, Path]) -> dict[str, Resource]:
    """
    Same as setup_api_services but uses an explicit credentials directory
    (expects client_secrets.json and reads/writes token.json there).
    """
    return _build_google_api_services(str(Path(credentials_dir).resolve()))
