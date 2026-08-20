#!/usr/bin/env python3
# transcriber/backends/openai_api.py

"""
OpenAI Whisper API Backend

Description:
Calls the OpenAI transcriptions endpoint (whisper-1 model) as a fallback or
primary backend. Handles HTTP 429 rate-limit responses with exponential backoff.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import logging
import time
from pathlib import Path
from typing import Optional

from transcriber.backends.base import TranscriptionBackend, TranscriptionResult


logger = logging.getLogger(__name__)

# Maximum audio file size accepted by the OpenAI API (25 MB)
_OPENAI_MAX_BYTES = 25 * 1024 * 1024

# Optional import
try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _openai = None  # type: ignore[assignment]


class OpenAIBackend(TranscriptionBackend):
    """
    Transcription backend using the OpenAI Whisper API.

    Implements rate-limit handling: on HTTP 429, waits for the Retry-After
    header duration (or falls back to exponential backoff) before retrying.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        """
        Args:
            api_key:     OpenAI API key.
            model:       Whisper model identifier (default: 'whisper-1').
            max_retries: Maximum number of retry attempts on rate-limit errors.
            base_delay:  Initial backoff delay in seconds.
            max_delay:   Maximum backoff delay in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    @property
    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        """Return True if the openai package is installed and an API key is set."""
        if not _OPENAI_AVAILABLE:
            logger.debug("openai package is not installed.")
            return False
        if not self.api_key:
            logger.debug("OpenAI API key is not configured.")
            return False
        return True

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file via the OpenAI API.

        Args:
            audio_path: Path to the audio file (max 25 MB).
            language:   ISO 639-1 language code, or None for auto-detect.

        Returns:
            TranscriptionResult populated from the API response.

        Raises:
            RuntimeError: If the openai package is missing, file too large,
                          or all retries are exhausted.
        """
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai package is not installed. Install with: pip install openai"
            )

        file_path = Path(audio_path)
        if not file_path.is_file():
            raise RuntimeError(f"Audio file does not exist: {audio_path}")

        file_size = file_path.stat().st_size
        if file_size > _OPENAI_MAX_BYTES:
            raise RuntimeError(
                f"File size {file_size} bytes exceeds OpenAI's 25 MB limit. "
                "Split the audio into smaller chunks first."
            )

        client = _openai.OpenAI(api_key=self.api_key)

        attempt = 0
        delay = self.base_delay

        while attempt <= self.max_retries:
            try:
                logger.debug(
                    "OpenAI API transcription attempt %d/%d: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    audio_path,
                )
                with file_path.open("rb") as audio_fh:
                    kwargs_api = {
                        "model": self.model,
                        "file": audio_fh,
                        "response_format": "verbose_json",
                    }
                    if language:
                        kwargs_api["language"] = language

                    response = client.audio.transcriptions.create(**kwargs_api)

                return self._parse_response(response, audio_path)

            except _openai.RateLimitError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OpenAI rate limit exceeded after {self.max_retries + 1} attempts."
                    ) from exc

                # Try to read Retry-After from the error headers
                retry_after = self._extract_retry_after(exc)
                wait_time = retry_after if retry_after else min(delay, self.max_delay)
                logger.warning(
                    "OpenAI rate limit hit. Waiting %.1fs before retry %d/%d...",
                    wait_time,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(wait_time)
                delay = min(delay * 2, self.max_delay)
                attempt += 1

            except _openai.APIStatusError as exc:
                raise RuntimeError(
                    f"OpenAI API error (status {exc.status_code}): {exc.message}"
                ) from exc

            except _openai.APIConnectionError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OpenAI connection error after {self.max_retries + 1} attempts: {exc}"
                    ) from exc
                logger.warning(
                    "OpenAI connection error. Retrying in %.1fs (%d/%d)...",
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(min(delay, self.max_delay))
                delay = min(delay * 2, self.max_delay)
                attempt += 1

        raise RuntimeError("OpenAI transcription failed: exhausted all retries.")

    @staticmethod
    def _extract_retry_after(exc: Exception) -> Optional[float]:
        """
        Attempt to extract Retry-After value from an openai exception.
        Returns None if not available.
        """
        try:
            headers = getattr(exc, "response", None)
            if headers is not None:
                retry_after_str = headers.headers.get("Retry-After")
                if retry_after_str:
                    return float(retry_after_str)
        except (AttributeError, ValueError):
            pass
        return None

    def _parse_response(
        self,
        response: object,
        source_file: str,
    ) -> TranscriptionResult:
        """Parse a verbose_json API response into a TranscriptionResult."""
        text = getattr(response, "text", "") or ""
        language = getattr(response, "language", None)
        duration = getattr(response, "duration", None)

        raw_segments = getattr(response, "segments", None) or []
        segments = []
        for seg in raw_segments:
            seg_dict = {
                "start": getattr(seg, "start", 0.0),
                "end": getattr(seg, "end", 0.0),
                "text": (getattr(seg, "text", "") or "").strip(),
            }
            segments.append(seg_dict)

        return TranscriptionResult(
            text=text.strip(),
            language=language,
            segments=segments,
            source_file=source_file,
            backend_name=self.name,
            duration=float(duration) if duration is not None else None,
        )
