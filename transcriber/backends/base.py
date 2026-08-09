#!/usr/bin/env python3
# transcriber/backends/base.py

"""
Base Backend Module

Description:
Abstract base class defining the interface that all transcription backends
must implement. Also defines the TranscriptionResult dataclass.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TranscriptionResult:
    """Normalized result returned by any transcription backend."""

    text: str
    language: Optional[str] = None
    # List of segment dicts with keys: start, end, text
    segments: List[dict] = field(default_factory=list)
    # Source chunk file path
    source_file: Optional[str] = None
    # Backend that produced this result
    backend_name: Optional[str] = None
    # Duration in seconds, if known
    duration: Optional[float] = None
    # True if this result was produced by translating another transcript
    translated: bool = False
    # Original source language when translated (ISO 639-1), else None
    source_language: Optional[str] = None

    def is_empty(self) -> bool:
        """Return True if the transcription text is empty or whitespace-only."""
        return not self.text.strip()


class TranscriptionBackend(ABC):
    """
    Abstract base class for all transcription backends.

    Subclasses must implement:
      - transcribe(audio_path, language, **kwargs) -> TranscriptionResult
      - is_available() -> bool
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (e.g. 'whisper_cpp')."""

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check whether this backend is usable in the current environment.

        Returns:
            True if the backend can be used, False otherwise.
        """

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe the audio file at the given path.

        Args:
            audio_path: Absolute path to the audio file.
            language:   ISO 639-1 language code (e.g. 'fr', 'en'), or None for auto-detect.
            **kwargs:   Backend-specific keyword arguments.

        Returns:
            TranscriptionResult with at minimum a non-empty 'text' field.

        Raises:
            RuntimeError: On unrecoverable backend errors.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} available={self.is_available()}>"
