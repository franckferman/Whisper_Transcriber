#!/usr/bin/env python3
# transcriber/managers/transcription.py

"""
Transcription Manager Module

Description:
Central orchestrator that wires together:
- VideoProcessor  (download + chunk splitting)
- Transcription backends (with retry + fallback)
- ThreadPoolExecutor for parallel chunk processing
- OutputFormatter for writing results

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from transcriber.backends.base import TranscriptionBackend, TranscriptionResult
from transcriber.backends.whisper_cpp import WhisperCppBackend
from transcriber.backends.faster_whisper import FasterWhisperBackend
from transcriber.backends.openai_api import OpenAIBackend
from transcriber.backends.mega_asr import MegaAsrBackend
from transcriber.config import TranscriptionConfig
from transcriber.formatters.output import OutputFormatter
from transcriber.processors.translate import LocalTranslator, TranslationError
from transcriber.processors.video import VideoProcessor
from transcriber.processors import word_timestamps as wt

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    _tqdm = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


def _make_backend(config: TranscriptionConfig, backend_name: str) -> TranscriptionBackend:
    """
    Instantiate a backend by name using settings from config.

    Args:
        config:       Active TranscriptionConfig.
        backend_name: One of 'whisper_cpp', 'faster_whisper', 'openai', 'mega_asr'.

    Returns:
        Configured TranscriptionBackend instance.

    Raises:
        ValueError: If backend_name is unrecognised.
    """
    if backend_name == "whisper_cpp":
        return WhisperCppBackend(
            binary_path=config.whisper_cpp_binary,
            model_path=config.whisper_cpp_model,
            extra_args=config.whisper_cpp_extra_args,
        )
    if backend_name == "faster_whisper":
        return FasterWhisperBackend(
            model_size=config.faster_whisper_model,
            device=config.faster_whisper_device,
            compute_type=config.faster_whisper_compute_type,
        )
    if backend_name == "openai":
        return OpenAIBackend(
            api_key=config.openai_api_key or "",
            model=config.openai_model,
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )
    if backend_name == "mega_asr":
        return MegaAsrBackend(
            repo_dir=config.mega_asr_repo_dir,
            ckpt_dir=config.mega_asr_ckpt_dir,
            device_map=config.mega_asr_device_map,
            allow_cpu=config.mega_asr_allow_cpu,
            force_lora=config.mega_asr_force_lora,
        )
    raise ValueError(f"Unknown backend name: {backend_name!r}")


class TranscriptionManager:
    """
    Orchestrates the full transcription pipeline:

    1. Resolve input (local / YouTube / HTTP) via VideoProcessor.
    2. Split into fixed-duration chunks.
    3. Transcribe each chunk in parallel using ThreadPoolExecutor.
       - Retry on failure with exponential backoff.
       - Automatic fallback to a secondary backend if the primary fails all retries.
    4. Merge chunk results.
    5. Write output files via OutputFormatter.
    """

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self.formatter = OutputFormatter(
            output_dir=config.output_dir,
            output_prefix=config.output_prefix,
        )
        self.video_processor = VideoProcessor(
            temp_dir=config.temp_dir,
            ytdlp_format=config.ytdlp_format,
            ytdlp_output_template=config.ytdlp_output_template,
        )
        # In translation-only mode there is no audio, so no backend is built.
        self._translate_only = bool(config.translate_text_input)
        if self._translate_only:
            self._primary_backend = None  # type: ignore[assignment]
            self._fallback_backend = None
        else:
            self._primary_backend = _make_backend(config, config.backend)
            self._fallback_backend: Optional[TranscriptionBackend] = (
                _make_backend(config, config.fallback_backend)
                if config.fallback_backend
                else None
            )

        # Native word timestamps: only the backend-driven provider asks the
        # backend to emit words per chunk. Premium providers align post-merge.
        self._native_words = bool(
            config.word_timestamps
            and wt.requires_backend_words(config.word_timestamps_provider)
        )

        # Track temp directories created by this manager
        self._temp_dirs: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full transcription pipeline.

        In dry-run mode, logs all actions but performs no actual transcription
        or file writes.

        Raises:
            ValueError:       On invalid configuration.
            FileNotFoundError: If the input file does not exist.
            RuntimeError:     On unrecoverable errors.
        """
        self.config.validate()

        # Translation-only mode: no audio, just translate an existing text file.
        if self._translate_only:
            self._run_translate_only()
            return

        source = self.config.input_url or self.config.input_file
        assert source is not None  # validate() guarantees this

        logger.info("Starting transcription pipeline. Source: %s", source)
        logger.info("Primary backend: %s", self.config.backend)
        if self.config.fallback_backend:
            logger.info("Fallback backend: %s", self.config.fallback_backend)
        logger.info("Output formats: %s", self.config.output_formats)
        logger.info("Workers: %d", self.config.workers)
        if self.config.translate_to:
            logger.info("Translation target: %s", self.config.translate_to)

        if self.config.dry_run:
            logger.info("[DRY-RUN] Would process: %s", source)
            logger.info("[DRY-RUN] Backend: %s", self.config.backend)
            logger.info("[DRY-RUN] Chunk duration: %ds", self.config.chunk_duration_seconds)
            if self.config.translate_to:
                logger.info(
                    "[DRY-RUN] Would translate transcript to: %s",
                    self.config.translate_to,
                )
            logger.info("[DRY-RUN] No files will be written.")
            return

        try:
            local_path = self.video_processor.resolve_to_local(source)
            logger.info("Resolved to local path: %s", local_path)

            chunks = self._prepare_chunks(local_path)
            logger.info("Processing %d chunk(s)...", len(chunks))

            results = self._transcribe_chunks_parallel(chunks)

            merged = self._merge_results(results)
            logger.info(
                "Transcription complete. Total characters: %d", len(merged.text)
            )

            # Premium word-timestamp providers align the merged transcript over
            # the full audio (the native provider already filled words per chunk).
            if (
                self.config.word_timestamps
                and not wt.requires_backend_words(self.config.word_timestamps_provider)
            ):
                merged = wt.align_result(
                    merged,
                    local_path,
                    self.config.word_timestamps_provider,
                    language=self.config.language or merged.language,
                )

            output_paths = self.formatter.write(merged, self.config.output_formats)
            for path in output_paths:
                logger.info("Output written: %s", path)

            # Optional translation stage: keep the original, add a translated copy.
            if self.config.translate_to:
                self._translate_and_write(merged)

        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Internal: translation
    # ------------------------------------------------------------------

    def _build_translator(self) -> LocalTranslator:
        """Construct a LocalTranslator from the current config."""
        return LocalTranslator(
            allow_download=self.config.translate_allow_download,
            package_path=self.config.translate_package_path,
        )

    def _translated_prefix(self) -> str:
        """Output prefix for translated files: '{prefix}.{target_lang}'."""
        return f"{self.config.output_prefix}.{self.config.translate_to}"

    def _translate_and_write(self, merged: TranscriptionResult) -> None:
        """
        Translate an already-written transcript and write the translated copy.

        A missing optional dependency or a translation error is logged and
        skipped -- the original transcript has already been written, so a
        translation failure never loses the primary result.
        """
        translator = self._build_translator()
        if not translator.is_available():
            logger.warning(
                "Translation requested (translate_to=%s) but 'argostranslate' "
                "is not installed; skipping. The original transcript was "
                "written. Install the optional translation extra to enable it.",
                self.config.translate_to,
            )
            return

        try:
            translated = translator.translate_result(
                merged,
                to_code=self.config.translate_to,  # type: ignore[arg-type]
                from_code=self.config.translate_from,
            )
        except TranslationError as exc:
            logger.error("Translation failed: %s", exc)
            return

        if translated is merged:
            # Source and target languages matched; nothing new to write.
            return

        translated_formatter = OutputFormatter(
            output_dir=self.config.output_dir,
            output_prefix=self._translated_prefix(),
        )
        paths = translated_formatter.write(translated, self.config.output_formats)
        for path in paths:
            logger.info("Translated output written: %s", path)

    def _run_translate_only(self) -> None:
        """
        Translate an existing text file with no transcription step.

        The source and target languages are both required (validated upstream),
        since a plain text file carries no detected language. Output goes to
        '{prefix}.{target}.{fmt}'.
        """
        text_path = Path(self.config.translate_text_input)  # type: ignore[arg-type]
        if not text_path.is_file():
            raise FileNotFoundError(f"Text file not found: {text_path}")

        source = self.config.translate_from
        target = self.config.translate_to
        logger.info(
            "Translation-only run: %s (%s -> %s)", text_path, source, target
        )

        if self.config.dry_run:
            logger.info(
                "[DRY-RUN] Would translate %s from %s to %s", text_path, source, target
            )
            logger.info("[DRY-RUN] No files will be written.")
            return

        translator = self._build_translator()
        if not translator.is_available():
            raise RuntimeError(
                "Translation requires the optional 'argostranslate' package, "
                "which is not installed."
            )

        text = text_path.read_text(encoding="utf-8")
        result = TranscriptionResult(
            text=text,
            language=source,
            source_file=str(text_path),
            backend_name="(translate-only)",
        )
        translated = translator.translate_result(
            result, to_code=target, from_code=source  # type: ignore[arg-type]
        )

        formatter = OutputFormatter(
            output_dir=self.config.output_dir,
            output_prefix=self._translated_prefix(),
        )
        paths = formatter.write(translated, self.config.output_formats)
        for path in paths:
            logger.info("Translated output written: %s", path)

    # ------------------------------------------------------------------
    # Internal: chunk preparation
    # ------------------------------------------------------------------

    def _prepare_chunks(self, local_path: str) -> List[str]:
        """
        Split the input file into chunks.

        If the file is short enough (single chunk), return it as-is wrapped
        in a list to avoid unnecessary splitting overhead.

        Args:
            local_path: Absolute path to the local audio/video file.

        Returns:
            List of absolute paths to chunk files.
        """
        chunks_dir = tempfile.mkdtemp(dir=self.config.temp_dir, prefix="wt_chunks_")
        self._temp_dirs.append(chunks_dir)

        # Chunks come out as 16 kHz mono WAV regardless of backend -- that's what
        # whisper.cpp requires, and faster-whisper/openai accept it too.
        chunks = self.video_processor.split_into_chunks(
            input_file=local_path,
            chunk_duration=self.config.chunk_duration_seconds,
            output_dir=chunks_dir,
            extension=".wav",
        )

        if not chunks:
            # Fallback: treat the original file as a single chunk
            logger.warning("No chunks produced by splitter; using original file as single chunk.")
            chunks = [local_path]

        return chunks

    # ------------------------------------------------------------------
    # Internal: parallel transcription
    # ------------------------------------------------------------------

    def _transcribe_chunks_parallel(
        self, chunks: List[str]
    ) -> List[TranscriptionResult]:
        """
        Transcribe all chunks using ThreadPoolExecutor.

        Args:
            chunks: List of audio file paths.

        Returns:
            List of TranscriptionResult objects in the same order as chunks.
        """
        results: List[Optional[TranscriptionResult]] = [None] * len(chunks)

        progress = None
        if _TQDM_AVAILABLE:
            progress = _tqdm(
                total=len(chunks),
                desc="Transcribing",
                unit="chunk",
                dynamic_ncols=True,
            )

        def transcribe_one(index: int, chunk_path: str) -> Tuple[int, TranscriptionResult]:
            result = self._transcribe_with_retry_and_fallback(chunk_path)
            return index, result

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            future_to_index = {
                executor.submit(transcribe_one, idx, path): idx
                for idx, path in enumerate(chunks)
            }

            for future in as_completed(future_to_index):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as exc:
                    chunk_idx = future_to_index[future]
                    logger.error(
                        "Chunk %d failed permanently: %s", chunk_idx, exc
                    )
                    # Insert empty result to maintain ordering
                    results[chunk_idx] = TranscriptionResult(
                        text="",
                        source_file=chunks[chunk_idx],
                        backend_name="(failed)",
                    )
                finally:
                    if progress is not None:
                        progress.update(1)

        if progress is not None:
            progress.close()

        # Filter out None entries (should not occur, but type-safe)
        return [r for r in results if r is not None]

    def _transcribe_with_retry_and_fallback(
        self, audio_path: str
    ) -> TranscriptionResult:
        """
        Attempt transcription with the primary backend, retrying on failure.
        If all retries fail, attempt with the fallback backend (if configured).

        Args:
            audio_path: Path to an audio chunk.

        Returns:
            TranscriptionResult from whichever backend succeeded.

        Raises:
            RuntimeError: If both primary and fallback fail.
        """
        last_exc: Optional[Exception] = None
        delay = self.config.retry_base_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(
                    "Transcribing %s with %s (attempt %d/%d)",
                    Path(audio_path).name,
                    self._primary_backend.name,
                    attempt + 1,
                    self.config.max_retries + 1,
                )
                return self._primary_backend.transcribe(
                    audio_path,
                    language=self.config.language,
                    word_timestamps=self._native_words,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    logger.warning(
                        "Backend '%s' failed for %s (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        self._primary_backend.name,
                        Path(audio_path).name,
                        attempt + 1,
                        self.config.max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, self.config.retry_max_delay)
                else:
                    logger.error(
                        "Backend '%s' exhausted all retries for %s: %s",
                        self._primary_backend.name,
                        Path(audio_path).name,
                        exc,
                    )

        # All primary retries exhausted - try fallback
        if self._fallback_backend is not None:
            logger.info(
                "Attempting fallback backend '%s' for %s",
                self._fallback_backend.name,
                Path(audio_path).name,
            )
            try:
                return self._fallback_backend.transcribe(
                    audio_path,
                    language=self.config.language,
                    word_timestamps=self._native_words,
                )
            except Exception as fb_exc:
                raise RuntimeError(
                    f"Both primary backend '{self._primary_backend.name}' and "
                    f"fallback '{self._fallback_backend.name}' failed for "
                    f"{audio_path}. "
                    f"Primary error: {last_exc}. "
                    f"Fallback error: {fb_exc}"
                ) from fb_exc

        raise RuntimeError(
            f"Primary backend '{self._primary_backend.name}' failed for {audio_path} "
            f"after {self.config.max_retries + 1} attempt(s): {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Internal: merge + cleanup
    # ------------------------------------------------------------------

    def _merge_results(self, results: List[TranscriptionResult]) -> TranscriptionResult:
        """
        Merge multiple chunk results into a single TranscriptionResult.

        Concatenates texts and flattens segments, adjusting timestamps
        so that each chunk's segments follow the previous chunk in time.

        Args:
            results: Ordered list of per-chunk TranscriptionResult objects.

        Returns:
            Merged TranscriptionResult.
        """
        combined_texts: List[str] = []
        combined_segments: List[dict] = []
        time_offset = 0.0
        detected_language: Optional[str] = None
        total_duration = 0.0

        for result in results:
            if result.text.strip():
                combined_texts.append(result.text.strip())

            for seg in result.segments:
                adjusted = dict(seg)
                adjusted["start"] = seg.get("start", 0.0) + time_offset
                adjusted["end"] = seg.get("end", 0.0) + time_offset
                # Shift nested word timings by the same chunk offset.
                if seg.get("words"):
                    adjusted["words"] = [
                        {
                            **w,
                            "start": w.get("start", 0.0) + time_offset,
                            "end": w.get("end", 0.0) + time_offset,
                        }
                        for w in seg["words"]
                    ]
                combined_segments.append(adjusted)

            if result.language and not detected_language:
                detected_language = result.language

            # Advance time offset by duration of this chunk
            if result.duration is not None:
                time_offset += result.duration
            elif result.segments:
                last_end = max(
                    seg.get("end", 0.0) for seg in result.segments
                )
                time_offset += last_end

            if result.duration is not None:
                total_duration += result.duration

        return TranscriptionResult(
            text=" ".join(combined_texts),
            language=detected_language,
            segments=combined_segments,
            backend_name=results[0].backend_name if results else None,
            duration=total_duration if total_duration > 0 else None,
        )

    def _cleanup(self) -> None:
        """Remove all temporary directories created during this session."""
        for tmp in self._temp_dirs:
            if Path(tmp).exists():
                try:
                    shutil.rmtree(tmp)
                    logger.debug("Removed temp dir: %s", tmp)
                except OSError as exc:
                    logger.warning("Failed to remove temp dir %s: %s", tmp, exc)
        self._temp_dirs.clear()
