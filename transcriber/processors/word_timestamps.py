#!/usr/bin/env python3
# transcriber/processors/word_timestamps.py

"""
Word-Level Timestamp Providers

Description:
Opt-in, additive word-level timestamps. A provider decides where the per-word
timing comes from:

  - 'native'    -- the transcription backend emits words itself. faster-whisper
                   (word_timestamps=True) and whisper.cpp (--output-json-full)
                   both do this; nothing is needed here. Light, no extra deps.
  - 'stable_ts' -- post-hoc forced alignment of the transcript over the audio via
                   stable-ts. Tighter word timing than the native path.
  - 'whisperx'  -- post-hoc forced alignment via whisperX (wav2vec2). Highest
                   alignment accuracy; heaviest (per-language alignment models).

Premium providers (stable_ts / whisperx) are optional extras: their imports are
lazy and, if the package is absent, alignment degrades gracefully to whatever
timing is already present (a warning is logged, the run still succeeds).

Words are stored inside each segment dict under a 'words' key:
    {"word": str, "start": float, "end": float, "probability": float | None}

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import logging
from typing import List, Optional

from transcriber.backends.base import TranscriptionResult


logger = logging.getLogger(__name__)

NATIVE = "native"
STABLE_TS = "stable_ts"
WHISPERX = "whisperx"


def requires_backend_words(provider: str) -> bool:
    """
    True if the backend itself must produce the words (the native provider).

    For premium providers the backend transcribes normally and a separate
    alignment pass fills the words afterwards, so the backend is not asked to.
    """
    return provider == NATIVE


def result_has_words(result: TranscriptionResult) -> bool:
    """True if at least one segment carries a non-empty 'words' list."""
    return any(seg.get("words") for seg in result.segments)


def align_result(
    result: TranscriptionResult,
    audio_path: str,
    provider: str,
    language: Optional[str] = None,
) -> TranscriptionResult:
    """
    Attach word-level timing to ``result`` using the given premium provider.

    The native provider is a no-op here (the backend already filled words).
    A missing optional dependency logs a warning and returns ``result``
    unchanged -- alignment never breaks a run.
    """
    if provider == NATIVE:
        return result
    if provider == STABLE_TS:
        return _StableTsAligner(language).align(result, audio_path)
    if provider == WHISPERX:
        return _WhisperXAligner(language).align(result, audio_path)
    logger.warning("Unknown word_timestamps provider '%s'; skipping.", provider)
    return result


class _StableTsAligner:
    """Forced alignment via stable-ts (MIT)."""

    def __init__(self, language: Optional[str] = None, model_size: str = "base") -> None:
        self.language = language
        self.model_size = model_size

    @staticmethod
    def is_available() -> bool:
        try:
            import stable_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def align(
        self, result: TranscriptionResult, audio_path: str
    ) -> TranscriptionResult:
        if not self.is_available():
            logger.warning(
                "word_timestamps provider 'stable_ts' requested but stable-ts is "
                "not installed; keeping existing timing. Install the align extra "
                "to enable it."
            )
            return result
        import stable_whisper  # lazy, heavy (pulls torch)

        logger.info("Aligning words with stable-ts (%s)...", self.model_size)
        model = stable_whisper.load_model(self.model_size)
        aligned = model.align(audio_path, result.text, language=self.language)

        segments: List[dict] = []
        for seg in aligned.segments:
            entry = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": [
                    {
                        "word": w.word,
                        "start": float(w.start),
                        "end": float(w.end),
                        "probability": getattr(w, "probability", None),
                    }
                    for w in (seg.words or [])
                ],
            }
            segments.append(entry)

        if not segments:
            logger.warning("stable-ts alignment produced no segments; keeping original.")
            return result

        return _with_segments(result, segments)


class _WhisperXAligner:
    """Forced alignment via whisperX (wav2vec2, BSD-2)."""

    def __init__(self, language: Optional[str] = None, device: str = "cpu") -> None:
        self.language = language
        self.device = device

    @staticmethod
    def is_available() -> bool:
        try:
            import whisperx  # noqa: F401
        except ImportError:
            return False
        return True

    def align(
        self, result: TranscriptionResult, audio_path: str
    ) -> TranscriptionResult:
        if not self.is_available():
            logger.warning(
                "word_timestamps provider 'whisperx' requested but whisperX is "
                "not installed; keeping existing timing. Install the align extra "
                "to enable it."
            )
            return result
        lang = self.language or result.language
        if not lang:
            logger.warning(
                "whisperX alignment needs a language; none known. Keeping timing."
            )
            return result
        import whisperx  # lazy, heavy

        logger.info("Aligning words with whisperX (lang=%s)...", lang)
        audio = whisperx.load_audio(audio_path)
        model_a, metadata = whisperx.load_align_model(
            language_code=lang, device=self.device
        )
        # whisperX expects segments as [{"start", "end", "text"}].
        in_segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.segments
        ]
        aligned = whisperx.align(
            in_segments, model_a, metadata, audio, self.device,
            return_char_alignments=False,
        )

        segments: List[dict] = []
        for seg in aligned.get("segments", []):
            entry = {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")).strip(),
                "words": [
                    {
                        "word": w.get("word", ""),
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "probability": w.get("score"),
                    }
                    for w in seg.get("words", [])
                    if w.get("start") is not None and w.get("end") is not None
                ],
            }
            segments.append(entry)

        if not segments:
            logger.warning("whisperX alignment produced no segments; keeping original.")
            return result

        return _with_segments(result, segments)


def _with_segments(
    result: TranscriptionResult, segments: List[dict]
) -> TranscriptionResult:
    """Return a copy of ``result`` with its segments replaced (text rebuilt)."""
    text = " ".join(s["text"] for s in segments if s.get("text")).strip()
    return TranscriptionResult(
        text=text or result.text,
        language=result.language,
        segments=segments,
        source_file=result.source_file,
        backend_name=result.backend_name,
        duration=result.duration,
        translated=result.translated,
        source_language=result.source_language,
    )
