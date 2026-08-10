#!/usr/bin/env python3
# transcriber/backends/mega_asr.py

"""
Mega-ASR Backend Module

Description:
Optional fourth backend wrapping Mega-ASR (https://github.com/xzf-thu/Mega-ASR),
a robustness-oriented model built as a LoRA + audio-quality router on top of
Qwen3-ASR-1.7B. It targets heavily degraded audio (noise, far-field, echo,
recording artefacts) where whisper.cpp / faster-whisper tend to hallucinate,
drop utterances, or return empty output.

Two facts shape this integration:

1. The LoRA is trained on English and Chinese only (Voices-in-the-Wild-2M is
   en/zh). Its robustness gain does not transfer to other languages. So the
   LoRA is mounted ONLY for tuned languages; for the other ~28 languages the
   base Qwen3-ASR model runs instead. Languages outside Qwen3-ASR's set are
   rejected up front.

2. Mega-ASR's upstream router keys off *audio quality*, not language -- it would
   mount the en/zh LoRA on noisy French. We deliberately bypass that router and
   drive the LoRA decision from the language instead, via the explicit
   infer_with_lora / infer_without_lora entry points.

The MegaASR wrapper class lives in the Mega-ASR repository (it is not part of
the ``qwen-asr`` PyPI package), so this backend needs the path to a local clone
of that repository plus its downloaded checkpoints. All heavy imports are lazy;
without them, is_available() returns False and the backend stays inert.

Timestamps: v1 returns text only (no segments), so SRT/VTT fall back to a single
block. Per-segment timestamps would require Qwen3-ForcedAligner and are left for
a later iteration.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional

from transcriber.backends.base import TranscriptionBackend, TranscriptionResult


logger = logging.getLogger(__name__)

# The 30 languages Qwen3-ASR-1.7B (Mega-ASR's base model) recognises.
MEGA_SUPPORTED_LANGUAGES = frozenset({
    "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it",
    "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv",
    "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro",
})

# The languages the Mega-ASR LoRA was actually trained on (Voices-in-the-Wild-2M).
# Only here does mounting the LoRA improve robustness.
MEGA_TUNED_LANGUAGES = frozenset({"en", "zh"})

# Factory that builds the underlying MegaASR model. Injected in tests.
ModelFactory = Callable[[], object]


class MegaAsrBackend(TranscriptionBackend):
    """
    Transcription backend backed by Mega-ASR (Qwen3-ASR + robustness LoRA).

    The model is loaded lazily on first use and reused across chunks. Because
    the underlying wrapper mutates shared LoRA state on every call, inference is
    serialised behind a lock: this backend is effectively single-threaded even
    when whispr runs multiple workers.
    """

    def __init__(
        self,
        repo_dir: Optional[str] = None,
        ckpt_dir: Optional[str] = None,
        *,
        device_map: Optional[str] = None,
        allow_cpu: bool = False,
        force_lora: bool = False,
        quality_threshold: float = 0.5,
        model_factory: Optional[ModelFactory] = None,
    ) -> None:
        """
        Args:
            repo_dir:       Path to a local clone of xzf-thu/Mega-ASR (provides
                            the MegaASR wrapper class).
            ckpt_dir:       Checkpoint root; defaults to '<repo_dir>/ckpt/Mega-ASR'.
            device_map:     'cuda:0' | 'mps' | 'cpu' | None (transformers device_map).
            allow_cpu:      Must be True to run on CPU; a 1.7B model in
                            transformers on CPU is very slow, so it is opt-in.
            force_lora:     Mount the LoRA regardless of language (experimentation).
            quality_threshold: Passed through to the wrapper (unused while the
                            router is bypassed, kept for forward-compat).
            model_factory:  Optional zero-arg callable returning a MegaASR-like
                            object with infer_with_lora / infer_without_lora.
                            Injected in tests to avoid the heavy dependency.
        """
        self.repo_dir = repo_dir
        self.ckpt_dir = ckpt_dir
        self.device_map = device_map
        self.allow_cpu = allow_cpu
        self.force_lora = force_lora
        self.quality_threshold = quality_threshold

        self._model_factory = model_factory
        self._model = None            # lazy-loaded on first use
        self._use_lora: Optional[bool] = None  # decided once, on first transcribe
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "mega_asr"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Return True if Mega-ASR can be used here.

        An injected model factory is always usable. Otherwise this needs the
        'qwen-asr' package, a Mega-ASR repo clone providing the wrapper, and a
        checkpoint directory on disk.
        """
        if self._model_factory is not None:
            return True

        try:
            import qwen_asr  # noqa: F401
        except ImportError:
            logger.debug("mega_asr unavailable: 'qwen-asr' is not installed.")
            return False

        if not self.repo_dir:
            logger.debug("mega_asr unavailable: no Mega-ASR repo_dir configured.")
            return False

        wrapper = Path(self.repo_dir) / "src" / "MegaASR" / "model" / "megaASR.py"
        if not wrapper.is_file():
            logger.debug("mega_asr unavailable: wrapper not found at %s", wrapper)
            return False

        if not Path(self._resolved_ckpt_dir()).is_dir():
            logger.debug(
                "mega_asr unavailable: checkpoint dir %s missing.",
                self._resolved_ckpt_dir(),
            )
            return False

        return True

    def _resolved_ckpt_dir(self) -> str:
        """Checkpoint root, defaulting to '<repo_dir>/ckpt/Mega-ASR'."""
        if self.ckpt_dir:
            return self.ckpt_dir
        if self.repo_dir:
            return str(Path(self.repo_dir) / "ckpt" / "Mega-ASR")
        return "ckpt/Mega-ASR"

    # ------------------------------------------------------------------
    # Language / LoRA decision
    # ------------------------------------------------------------------

    def _decide_use_lora(self, language: Optional[str]) -> bool:
        """
        Decide once whether to mount the LoRA, from the requested language.

        - force_lora overrides everything.
        - A tuned language (en/zh) mounts the LoRA (the robustness gain lives here).
        - A supported-but-not-tuned language runs the base model (LoRA off) and
          warns, because the LoRA never saw that language.
        - An unknown/None language runs the base model (conservative default).

        Raises:
            RuntimeError: If a language is explicitly requested that Qwen3-ASR
                          does not support (defensive; also caught at config
                          validation).
        """
        if self.force_lora:
            logger.info("mega_asr: force_lora set -> LoRA mounted regardless of language.")
            return True

        if language is None:
            logger.info(
                "mega_asr: no language set -> running base Qwen3-ASR without the "
                "LoRA (the robustness LoRA is en/zh-only; enable it explicitly "
                "for tuned languages)."
            )
            return False

        if language not in MEGA_SUPPORTED_LANGUAGES:
            raise RuntimeError(
                f"mega_asr does not support language '{language}'. Supported: "
                f"{sorted(MEGA_SUPPORTED_LANGUAGES)}."
            )

        if language in MEGA_TUNED_LANGUAGES:
            logger.info("mega_asr: language '%s' is tuned -> LoRA mounted.", language)
            return True

        logger.warning(
            "mega_asr: the robustness LoRA is trained on %s only; language '%s' "
            "runs on base Qwen3-ASR without the LoRA.",
            sorted(MEGA_TUNED_LANGUAGES), language,
        )
        return False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Instantiate the MegaASR wrapper (or the injected factory) once."""
        if self._model is not None:
            return

        if self._model_factory is not None:
            self._model = self._model_factory()
            return

        device_map = self.device_map
        if device_map in (None, "cpu") and not self.allow_cpu:
            raise RuntimeError(
                "mega_asr would run on CPU, which is very slow for a 1.7B model. "
                "Set a GPU device_map (e.g. 'cuda:0'/'mps') or pass allow_cpu."
            )

        if not self.repo_dir:
            raise RuntimeError(
                "mega_asr requires 'mega_asr_repo_dir' pointing at a local clone "
                "of xzf-thu/Mega-ASR (it provides the MegaASR wrapper)."
            )

        # Make the repo's 'src' importable, then load its wrapper.
        import sys
        src_dir = str(Path(self.repo_dir) / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        try:
            from MegaASR.model.megaASR import MegaASR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import the MegaASR wrapper from {src_dir}. Ensure "
                f"mega_asr_repo_dir points at a Mega-ASR clone and its "
                f"dependencies (requirements-mega.txt) are installed."
            ) from exc

        ckpt = Path(self._resolved_ckpt_dir())
        logger.info("mega_asr: loading model from %s (device_map=%s)", ckpt, device_map)
        # routing_enabled=False: we bypass the audio-quality router and drive the
        # LoRA from the language via infer_with_lora / infer_without_lora.
        self._model = MegaASR(
            model_path=ckpt / "Qwen3-ASR-1.7B",
            lora_dir=ckpt / "mega-asr-merged",
            router_checkpoint=ckpt / "audio_quality_router" / "best_acc_model.safetensors",
            routing_enabled=False,
            quality_threshold=self.quality_threshold,
            device_map=device_map,
            backend="transformers",
        )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_text(result: object) -> str:
        """Normalise the wrapper's return value into a plain transcript string."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, (list, tuple)):
            return " ".join(str(r).strip() for r in result if r).strip()
        return str(result).strip()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe one audio chunk with Mega-ASR.

        The LoRA decision is made once (on the first call) from the language and
        then held fixed for the whole run, so parallel chunks never flip shared
        LoRA state mid-flight. Inference is serialised behind a lock.

        Returns a text-only result (no segments in v1).
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with self._lock:
            if self._use_lora is None:
                self._use_lora = self._decide_use_lora(language)
            self._load_model()

            assert self._model is not None
            # Let Qwen3-ASR do its own language identification; the whispr
            # language only gates the LoRA (Qwen expects language names, not
            # ISO codes, so passing None avoids a mismatch).
            if self._use_lora:
                raw = self._model.infer_with_lora(audio_path)
            else:
                raw = self._model.infer_without_lora(audio_path)

        text = self._coerce_text(raw)
        return TranscriptionResult(
            text=text,
            language=language,
            segments=[],
            source_file=audio_path,
            backend_name=self.name,
        )
