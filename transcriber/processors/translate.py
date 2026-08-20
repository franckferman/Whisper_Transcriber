#!/usr/bin/env python3
# transcriber/processors/translate.py

"""
Translation Processor Module

Description:
Fully local, offline machine translation of a TranscriptionResult from its
source language into a target language. Segment timestamps are preserved so
SRT/VTT output stays aligned after translation.

Translation is powered by Argos Translate (OPUS-MT models running on the same
CTranslate2 engine that faster-whisper already uses). The dependency is
optional and imported lazily: if it is not installed, is_available() returns
False and the caller is expected to degrade gracefully -- exactly like the
optional transcription backends.

No online service is contacted at translation time. Language packages are
installed once (either from a local .argosmodel file for a 100% offline setup,
or downloaded from the Argos package index when explicitly allowed), then all
inference runs on-device.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import logging
from typing import Callable, List, Optional

from transcriber.backends.base import TranscriptionResult


logger = logging.getLogger(__name__)

# Type of the underlying translate primitive: (text, from_code, to_code) -> text
TranslateFn = Callable[[str, str, str], str]


class TranslationError(RuntimeError):
    """Raised when a translation cannot be performed."""


class LocalTranslator:
    """
    Translate transcription results locally via Argos Translate.

    The translate primitive can be injected (``translate_fn``) to make the
    orchestration logic testable without the heavy optional dependency. When
    no primitive is injected, it is resolved lazily from ``argostranslate``.
    """

    def __init__(
        self,
        translate_fn: Optional[TranslateFn] = None,
        *,
        allow_download: bool = True,
        package_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            translate_fn:   Optional (text, from_code, to_code) -> text callable.
                            Injected mainly for testing; defaults to Argos.
            allow_download: If True, missing language pairs may be fetched from
                            the Argos package index (one-time, online). If False,
                            only locally installed pairs (or ``package_path``)
                            are used -- a fully offline setup.
            package_path:   Optional path to a local ``.argosmodel`` file to
                            install the language pair from, for offline setups.
        """
        self._injected_fn = translate_fn
        self._translate_fn: Optional[TranslateFn] = translate_fn
        self.allow_download = allow_download
        self.package_path = package_path
        # Track pairs we have already ensured, to avoid repeated package scans.
        self._ready_pairs: set = set()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Return True if this translator can actually translate.

        An injected primitive (tests / custom engines) is always usable;
        otherwise availability depends on the optional ``argostranslate``
        package being importable.
        """
        if self._injected_fn is not None:
            return True
        try:
            import argostranslate.translate  # noqa: F401
        except ImportError:
            return False
        return True

    def _resolve_translate_fn(self) -> TranslateFn:
        """Return the translate primitive, importing Argos on first use."""
        if self._translate_fn is not None:
            return self._translate_fn
        try:
            import argostranslate.translate as _argos_translate
        except ImportError as exc:  # pragma: no cover - exercised via is_available
            raise TranslationError(
                "Local translation requires the 'argostranslate' package. "
                "Install it (and its language models) to enable --translate-to. "
                "It is an optional extra and is not part of whispr's core "
                "dependencies."
            ) from exc

        # The convenience translate(q, from, to) exists in argostranslate 1.x but
        # was dropped on newer trees; fall back to the stable object API so we
        # keep working across versions.
        if hasattr(_argos_translate, "translate"):
            self._translate_fn = _argos_translate.translate
        elif hasattr(_argos_translate, "get_translation_from_codes"):
            def _fn(text: str, from_code: str, to_code: str) -> str:
                return _argos_translate.get_translation_from_codes(
                    from_code, to_code
                ).translate(text)
            self._translate_fn = _fn
        else:
            def _fn(text: str, from_code: str, to_code: str) -> str:
                by_code = {
                    lang.code: lang
                    for lang in _argos_translate.get_installed_languages()
                }
                return by_code[from_code].get_translation(
                    by_code[to_code]
                ).translate(text)
            self._translate_fn = _fn
        return self._translate_fn

    # ------------------------------------------------------------------
    # Language pair management
    # ------------------------------------------------------------------

    def ensure_pair(self, from_code: str, to_code: str) -> None:
        """
        Make sure the from->to language pair is installed.

        When a primitive was injected (tests / custom engines), package
        management is skipped entirely. Otherwise the Argos package registry is
        consulted: an already-installed pair is a no-op; a missing one is
        installed from ``package_path`` if given, else downloaded when
        ``allow_download`` is True.

        Raises:
            TranslationError: If the pair is unavailable and cannot be provisioned.
        """
        pair = (from_code, to_code)
        if pair in self._ready_pairs:
            return

        # An injected primitive is assumed to already handle any pair.
        if self._injected_fn is not None:
            self._ready_pairs.add(pair)
            return

        try:
            import argostranslate.package as _argos_package
            import argostranslate.translate as _argos_translate
        except ImportError as exc:  # pragma: no cover
            raise TranslationError(
                "Local translation requires the 'argostranslate' package."
            ) from exc

        # Is the pair already installed? Use the stable Language object API:
        # find both languages by code and check a translation exists between them.
        by_code = {
            lang.code: lang for lang in _argos_translate.get_installed_languages()
        }
        src_lang = by_code.get(from_code)
        dst_lang = by_code.get(to_code)
        if (
            src_lang is not None
            and dst_lang is not None
            and src_lang.get_translation(dst_lang) is not None
        ):
            self._ready_pairs.add(pair)
            return

        # Install from a local package file if provided (offline path).
        if self.package_path:
            logger.info(
                "Installing %s->%s translation model from %s",
                from_code, to_code, self.package_path,
            )
            _argos_package.install_from_path(self.package_path)
            self._ready_pairs.add(pair)
            return

        if not self.allow_download:
            raise TranslationError(
                f"No local translation model for {from_code}->{to_code} and "
                f"downloads are disabled. Provide a local .argosmodel via "
                f"'translate_package_path' or allow downloads."
            )

        logger.info(
            "Downloading %s->%s translation model from the Argos index...",
            from_code, to_code,
        )
        _argos_package.update_package_index()
        available = _argos_package.get_available_packages()
        match = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if match is None:
            raise TranslationError(
                f"No Argos translation model is available for "
                f"{from_code}->{to_code}."
            )
        _argos_package.install_from_path(match.download())
        self._ready_pairs.add(pair)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate_text(self, text: str, from_code: str, to_code: str) -> str:
        """
        Translate a single block of text.

        Empty / whitespace-only input is returned unchanged. A no-op pair
        (from == to) short-circuits without touching the engine.
        """
        if not text or not text.strip():
            return text
        if from_code == to_code:
            return text
        self.ensure_pair(from_code, to_code)
        translate = self._resolve_translate_fn()
        return translate(text, from_code, to_code)

    def translate_segments(
        self, segments: List[dict], from_code: str, to_code: str
    ) -> List[dict]:
        """
        Translate each segment's text, preserving its start/end timestamps.

        Segments are translated individually to keep the timeline aligned for
        SRT/VTT. Note this trades a little quality for alignment: an engine
        translating one segment at a time has less surrounding context than one
        seeing the whole paragraph.
        """
        if from_code == to_code:
            return [dict(seg) for seg in segments]

        self.ensure_pair(from_code, to_code)
        translate = self._resolve_translate_fn()

        translated: List[dict] = []
        for seg in segments:
            new_seg = dict(seg)
            original = str(seg.get("text", ""))
            if original.strip():
                new_seg["text"] = translate(original, from_code, to_code)
            translated.append(new_seg)
        return translated

    def translate_result(
        self,
        result: TranscriptionResult,
        to_code: str,
        from_code: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Return a new TranscriptionResult translated into ``to_code``.

        The source language is ``from_code`` when given, else the result's own
        detected ``language``. The full text is translated once for the best
        context, and segments are translated individually to preserve timing.

        Args:
            result:    The transcription to translate (left unmodified).
            to_code:   Target ISO 639-1 language code.
            from_code: Source ISO 639-1 code; defaults to result.language.

        Returns:
            A new TranscriptionResult with translated text/segments, its
            ``language`` set to the target, and provenance recorded in
            ``source_language`` / ``translated``.

        Raises:
            TranslationError: If the source language is unknown or the pair
                              cannot be provisioned.
        """
        source = from_code or result.language
        if not source:
            raise TranslationError(
                "Cannot translate: source language is unknown. Pass an explicit "
                "source (e.g. --translate-from en) or transcribe with a known "
                "--language."
            )

        if source == to_code:
            logger.info(
                "Translation source and target are both '%s'; skipping.", to_code
            )
            return result

        logger.info("Translating transcript %s -> %s ...", source, to_code)

        translated_text = self.translate_text(result.text, source, to_code)
        translated_segments = self.translate_segments(
            result.segments, source, to_code
        )

        return TranscriptionResult(
            text=translated_text,
            language=to_code,
            segments=translated_segments,
            source_file=result.source_file,
            backend_name=result.backend_name,
            duration=result.duration,
            translated=True,
            source_language=source,
        )
