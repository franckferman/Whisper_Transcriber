#!/usr/bin/env python3
# transcriber/backends/__init__.py

"""
Backends Package

Provides the abstract base class and concrete backend implementations
for audio transcription.
"""

from transcriber.backends.base import TranscriptionBackend, TranscriptionResult

__all__ = ["TranscriptionBackend", "TranscriptionResult"]
