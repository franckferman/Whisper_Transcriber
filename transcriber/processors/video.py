#!/usr/bin/env python3
# transcriber/processors/video.py

"""
Video Processor Module

Description:
Handles downloading video/audio from local files, YouTube URLs, and generic
HTTP URLs. Also handles chunking/splitting long media into fixed-duration
segments. Reuses and extends the audio splitting logic from audio_splitter repo.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Optional yt-dlp import
try:
    import yt_dlp as _yt_dlp
    _YTDLP_AVAILABLE = True
except ImportError:
    _YTDLP_AVAILABLE = False
    _yt_dlp = None  # type: ignore[assignment]

# Optional requests for plain HTTP downloads
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    _requests = None  # type: ignore[assignment]

# Optional ffmpeg-python
try:
    import ffmpeg as _ffmpeg
    _FFMPEG_AVAILABLE = True
except ImportError:
    _FFMPEG_AVAILABLE = False
    _ffmpeg = None  # type: ignore[assignment]


def _is_youtube_url(url: str) -> bool:
    """Return True if the URL looks like a YouTube link."""
    parsed = urlparse(url)
    return parsed.hostname in {
        "www.youtube.com", "youtube.com", "youtu.be",
        "m.youtube.com", "music.youtube.com",
    }


def _is_http_url(url: str) -> bool:
    """Return True if the URL uses http or https scheme."""
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"}


class VideoProcessor:
    """
    Responsible for:
    1. Resolving the input (local file / YouTube URL / generic URL) to a local path.
    2. Splitting the resolved file into fixed-duration chunks.
    """

    def __init__(
        self,
        temp_dir: Optional[str] = None,
        ytdlp_format: str = "bestaudio/best",
        ytdlp_output_template: str = "%(title)s.%(ext)s",
    ) -> None:
        """
        Args:
            temp_dir:              Directory used for temporary files.
                                   If None, the OS default temp dir is used.
            ytdlp_format:          yt-dlp format selector string.
            ytdlp_output_template: yt-dlp output filename template.
        """
        self.temp_dir = temp_dir
        self.ytdlp_format = ytdlp_format
        self.ytdlp_output_template = ytdlp_output_template

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def resolve_to_local(self, source: str) -> str:
        """
        Download or copy the source to a local temporary file.

        Args:
            source: Local file path, YouTube URL, or generic HTTP URL.

        Returns:
            Absolute path to the local file.

        Raises:
            FileNotFoundError: If a local path does not exist.
            RuntimeError:      On download failures.
        """
        if _is_youtube_url(source) or (not Path(source).exists() and _is_http_url(source)):
            if _is_youtube_url(source):
                logger.info("Detected YouTube URL: %s", source)
                return self._download_youtube(source)
            else:
                logger.info("Detected HTTP URL: %s", source)
                return self._download_http(source)
        else:
            # Treat as local file
            local_path = Path(source)
            if not local_path.is_file():
                raise FileNotFoundError(f"Local file not found: {source}")
            logger.debug("Using local file: %s", source)
            return str(local_path.resolve())

    def split_into_chunks(
        self,
        input_file: str,
        chunk_duration: int = 600,
        output_dir: Optional[str] = None,
        extension: str = ".mp3",
    ) -> List[str]:
        """
        Split an audio/video file into fixed-duration chunks.
        Reuses the ffmpeg-based splitting logic from the audio_splitter repo.

        Args:
            input_file:     Path to the source audio/video file.
            chunk_duration: Duration of each chunk in seconds (default 600 = 10 min).
            output_dir:     Directory to write chunks into. Uses a temp dir if None.
            extension:      Output file extension (default .mp3).

        Returns:
            Sorted list of absolute paths to the generated chunk files.

        Raises:
            RuntimeError: If ffmpeg is not available or splitting fails.
        """
        if not _FFMPEG_AVAILABLE:
            raise RuntimeError(
                "ffmpeg-python is not installed. Install with: pip install ffmpeg-python"
            )

        if output_dir is None:
            output_dir = tempfile.mkdtemp(dir=self.temp_dir, prefix="chunks_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        output_pattern = os.path.join(output_dir, f"segment_%03d{extension}")

        file_path = Path(input_file)
        if not file_path.is_file():
            raise RuntimeError(f"Input file not found: {input_file}")

        # If the file is not a direct audio format, extract audio first
        supported_audio = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
        if file_path.suffix.lower() not in supported_audio:
            logger.info("Extracting audio from video file: %s", input_file)
            extracted = self._extract_audio(input_file, extension=".mp3")
            input_file = extracted

        logger.info(
            "Splitting '%s' into %ds chunks -> %s", input_file, chunk_duration, output_dir
        )

        try:
            (
                _ffmpeg
                .input(input_file)
                .output(
                    output_pattern,
                    f="segment",
                    segment_time=chunk_duration,
                    c="copy",
                    reset_timestamps=1,
                )
                .run(capture_stdout=True, capture_stderr=True)
            )
        except _ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "(no stderr)"
            raise RuntimeError(f"ffmpeg splitting failed:\n{stderr}") from exc

        chunks = sorted(
            str(p) for p in Path(output_dir).iterdir()
            if p.suffix.lower() == extension.lower()
        )
        logger.info("Created %d chunk(s).", len(chunks))
        return chunks

    def cleanup_temp(self, path: str) -> None:
        """Remove a temporary file or directory, ignoring errors."""
        target = Path(path)
        try:
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
        except OSError as exc:
            logger.warning("Could not remove temp path %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_youtube(self, url: str) -> str:
        """Download audio from a YouTube URL using yt-dlp."""
        if not _YTDLP_AVAILABLE:
            raise RuntimeError(
                "yt-dlp is not installed. Install with: pip install yt-dlp"
            )

        download_dir = tempfile.mkdtemp(dir=self.temp_dir, prefix="ytdlp_")
        output_template = os.path.join(download_dir, self.ytdlp_output_template)

        ydl_opts = {
            "format": self.ytdlp_format,
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }

        logger.info("Downloading from YouTube: %s", url)
        with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
            except Exception as exc:
                raise RuntimeError(f"yt-dlp download failed for {url}: {exc}") from exc

        # Find the downloaded file
        downloaded = list(Path(download_dir).iterdir())
        if not downloaded:
            raise RuntimeError(f"yt-dlp produced no output files for: {url}")

        # Prefer .mp3 if multiple files
        mp3_files = [f for f in downloaded if f.suffix.lower() == ".mp3"]
        chosen = mp3_files[0] if mp3_files else downloaded[0]
        logger.info("Downloaded to: %s", chosen)
        return str(chosen)

    def _download_http(self, url: str) -> str:
        """Download a file from a plain HTTP/HTTPS URL."""
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError(
                "requests is not installed. Install with: pip install requests"
            )

        parsed = urlparse(url)
        filename = Path(parsed.path).name or "downloaded_audio"
        download_dir = tempfile.mkdtemp(dir=self.temp_dir, prefix="http_dl_")
        dest = os.path.join(download_dir, filename)

        logger.info("Downloading from HTTP URL: %s -> %s", url, dest)
        try:
            response = _requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)
        except Exception as exc:
            raise RuntimeError(f"HTTP download failed for {url}: {exc}") from exc

        logger.info("Downloaded to: %s", dest)
        return dest

    def _extract_audio(self, input_file: str, extension: str = ".mp3") -> str:
        """Extract audio track from a video file using ffmpeg."""
        if not _FFMPEG_AVAILABLE:
            raise RuntimeError(
                "ffmpeg-python is not installed. Install with: pip install ffmpeg-python"
            )

        src = Path(input_file)
        dest = str(src.with_suffix(".extracted" + extension))
        logger.debug("Extracting audio: %s -> %s", input_file, dest)

        try:
            (
                _ffmpeg
                .input(input_file)
                .output(dest, acodec="libmp3lame", vn=None)
                .run(capture_stdout=True, capture_stderr=True)
            )
        except _ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "(no stderr)"
            raise RuntimeError(f"Audio extraction failed:\n{stderr}") from exc

        return dest
