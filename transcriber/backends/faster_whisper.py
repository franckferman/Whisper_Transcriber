#!/usr/bin/env python3
# transcriber/backends/faster_whisper.py

"""
Faster-Whisper Backend

Description:
Uses the faster-whisper Python package (CTranslate2-based) for transcription.
Import is guarded so the system works even if faster-whisper is not installed.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import logging
from pathlib import Path
from typing import Optional

from transcriber.backends.base import TranscriptionBackend, TranscriptionResult


logger = logging.getLogger(__name__)

# Optional import - graceful degradation if package absent
try:
    from faster_whisper import WhisperModel as _WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False
    _WhisperModel = None  # type: ignore[assignment,misc]


class FasterWhisperBackend(TranscriptionBackend):
    """
    Transcription backend using the faster-whisper Python library.

    faster-whisper is a reimplementation of OpenAI's Whisper using CTranslate2,
    offering significantly faster inference with lower memory usage.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        """
        Args:
            model_size:    Whisper model size (tiny, base, small, medium, large-v2, etc.).
            device:        Inference device ('cpu' or 'cuda').
            compute_type:  Quantization type ('int8', 'float16', 'float32', etc.).
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None  # Lazy-loaded on first use

    @property
    def name(self) -> str:
        return "faster_whisper"

    def is_available(self) -> bool:
        """Return True if the faster-whisper package is importable."""
        if not _FASTER_WHISPER_AVAILABLE:
            logger.debug("faster-whisper package is not installed.")
            return False
        return True

    def _load_model(self) -> None:
        """Load the WhisperModel if not already loaded."""
        if self._model is None:
            if not _FASTER_WHISPER_AVAILABLE:
                raise RuntimeError(
                    "faster-whisper is not installed. "
                    "Install it with: pip install faster-whisper"
                )
            logger.info(
                "Loading faster-whisper model '%s' on %s (%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = _WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("faster-whisper model loaded.")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using faster-whisper.

        Args:
            audio_path: Path to the audio file.
            language:   ISO 639-1 language code, or None for auto-detect.

        Returns:
            TranscriptionResult with full text and per-segment data.

        Raises:
            RuntimeError: If faster-whisper is unavailable or transcription fails.
        """
        if not Path(audio_path).is_file():
            raise RuntimeError(f"Audio file does not exist: {audio_path}")

        if not _FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            )

        self._load_model()

        transcribe_kwargs = {}
        if language:
            transcribe_kwargs["language"] = language

        logger.debug("faster-whisper transcribing: %s", audio_path)

        try:
            segments_iter, info = self._model.transcribe(  # type: ignore[union-attr]
                audio_path,
                beam_size=5,
                **transcribe_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"faster-whisper transcription failed for {audio_path}: {exc}"
            ) from exc

        segments = []
        texts = []

        for seg in segments_iter:
            text = seg.text.strip()
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
            })
            texts.append(text)

        detected_language = getattr(info, "language", None) or language

        return TranscriptionResult(
            text=" ".join(texts),
            language=detected_language,
            segments=segments,
            source_file=audio_path,
            backend_name=self.name,
            duration=getattr(info, "duration", None),
        )
