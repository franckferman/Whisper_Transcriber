#!/usr/bin/env python3
# transcriber/processors/audio.py

"""
Audio Processor Module

Description:
Audio format detection, validation, conversion to WAV/MP3, and basic
metadata probing (duration). Provides utilities needed by the transcription
pipeline before audio is handed off to a backend.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Optional ffmpeg-python
try:
    import ffmpeg as _ffmpeg
    _FFMPEG_AVAILABLE = True
except ImportError:
    _FFMPEG_AVAILABLE = False
    _ffmpeg = None  # type: ignore[assignment]


def probe_duration(file_path: str) -> Optional[float]:
    """
    Return the duration of an audio/video file in seconds.

    Uses ffprobe (part of FFmpeg) if available; falls back to pydub.
    Returns None if duration cannot be determined.

    Args:
        file_path: Path to the media file.

    Returns:
        Duration in seconds, or None.
    """
    # Try ffprobe first (more reliable, handles all formats)
    ffprobe = _find_ffprobe()
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                raw = result.stdout.strip()
                if raw and raw.lower() != "n/a":
                    return float(raw)
        except (subprocess.SubprocessError, ValueError) as exc:
            logger.debug("ffprobe duration probe failed: %s", exc)

    # Fallback: pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception as exc:
        logger.debug("pydub duration probe failed: %s", exc)

    return None


def convert_to_wav(
    input_path: str,
    output_dir: Optional[str] = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Convert an audio file to 16-bit PCM WAV (mono by default).
    This format is required by whisper.cpp.

    Args:
        input_path:  Source audio file.
        output_dir:  Directory for the output WAV. Uses temp dir if None.
        sample_rate: Target sample rate (whisper.cpp expects 16000 Hz).
        channels:    Number of channels (1 = mono, recommended for speech).

    Returns:
        Absolute path to the output WAV file.

    Raises:
        RuntimeError: If ffmpeg is unavailable or conversion fails.
    """
    if not _FFMPEG_AVAILABLE:
        raise RuntimeError(
            "ffmpeg-python is not installed. Install with: pip install ffmpeg-python"
        )

    src = Path(input_path)
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="wav_convert_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    dest = os.path.join(output_dir, src.stem + ".wav")
    logger.debug("Converting to WAV: %s -> %s", input_path, dest)

    try:
        (
            _ffmpeg
            .input(input_path)
            .output(
                dest,
                ar=sample_rate,
                ac=channels,
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except _ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "(no stderr)"
        raise RuntimeError(f"WAV conversion failed:\n{stderr}") from exc

    return dest


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _find_ffprobe() -> Optional[str]:
    """Return the path to ffprobe if it is on PATH, else None."""
    import shutil
    return shutil.which("ffprobe")
