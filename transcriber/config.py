#!/usr/bin/env python3
# transcriber/config.py

"""
Config Module

Description:
Configuration management with JSON file loading, CLI argument overrides,
and environment variable interpolation (${VAR_NAME} syntax in JSON values).

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import json
import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


logger = logging.getLogger(__name__)

# Regex to match ${VAR_NAME} placeholders
_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")

# ISO 639-1 codes are two lowercase letters (e.g. 'en', 'fr').
_ISO639_1_PATTERN = re.compile(r"^[a-z]{2}$")


def _is_iso639_1(code: str) -> bool:
    """Return True if ``code`` looks like an ISO 639-1 language code."""
    return bool(_ISO639_1_PATTERN.match(code))


def _interpolate_env(value: str) -> str:
    """Replace ${VAR_NAME} references in a string with environment variable values."""
    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            logger.warning("Environment variable '%s' is not set; keeping placeholder.", var_name)
            return match.group(0)
        return env_val

    return _ENV_PATTERN.sub(replace, value)


def _interpolate_dict(data: dict) -> dict:
    """Recursively interpolate environment variables in all string values of a dict."""
    result = {}
    for key, val in data.items():
        if isinstance(val, str):
            result[key] = _interpolate_env(val)
        elif isinstance(val, dict):
            result[key] = _interpolate_dict(val)
        elif isinstance(val, list):
            result[key] = [
                _interpolate_env(item) if isinstance(item, str) else item
                for item in val
            ]
        else:
            result[key] = val
    return result


@dataclass
class TranscriptionConfig:
    """Central configuration object for the transcription system."""

    # Input
    input_file: Optional[str] = None
    input_url: Optional[str] = None

    # Backend
    backend: str = "faster_whisper"
    fallback_backend: Optional[str] = None

    # whisper.cpp specific
    whisper_cpp_binary: str = "whisper"
    whisper_cpp_model: str = "models/ggml-base.bin"
    whisper_cpp_extra_args: List[str] = field(default_factory=list)

    # faster-whisper specific
    faster_whisper_model: str = "base"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"

    # OpenAI API specific
    openai_api_key: Optional[str] = None
    openai_model: str = "whisper-1"

    # Mega-ASR specific (optional 'mega' extra; robustness LoRA on Qwen3-ASR)
    mega_asr_repo_dir: Optional[str] = None   # path to a xzf-thu/Mega-ASR clone
    mega_asr_ckpt_dir: Optional[str] = None   # defaults to <repo>/ckpt/Mega-ASR
    mega_asr_device_map: Optional[str] = None  # 'cuda:0' | 'mps' | 'cpu' | None
    mega_asr_allow_cpu: bool = False          # opt-in: 1.7B on CPU is slow
    mega_asr_force_lora: bool = False         # mount LoRA regardless of language

    # Processing
    language: Optional[str] = None
    chunk_duration_seconds: int = 600
    workers: int = 2
    temp_dir: Optional[str] = None

    # Word-level timestamps (opt-in, additive). Provider selects the source:
    #   'native'    -- the backend's own word timing (faster_whisper / whisper.cpp)
    #   'stable_ts' -- post-hoc forced alignment via stable-ts (optional extra)
    #   'whisperx'  -- post-hoc forced alignment via whisperX (optional extra)
    word_timestamps: bool = False
    word_timestamps_provider: str = "native"

    # Output
    output_formats: List[str] = field(default_factory=lambda: ["txt"])
    output_dir: str = "."
    output_prefix: str = "transcript"

    # Translation (fully local, optional -- requires the 'argostranslate' extra)
    translate_to: Optional[str] = None       # target ISO 639-1 code; enables translation
    translate_from: Optional[str] = None     # source override; defaults to detected language
    translate_package_path: Optional[str] = None  # local .argosmodel for offline setups
    translate_allow_download: bool = True    # allow one-time model download from the Argos index
    translate_text_input: Optional[str] = None  # translate an existing text file, no transcription

    # Retry / resilience
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    # Misc
    dry_run: bool = False
    debug: bool = False
    log_file: Optional[str] = None

    # yt-dlp
    ytdlp_format: str = "bestaudio/best"
    ytdlp_output_template: str = "%(title)s.%(ext)s"

    def validate(self) -> None:
        """
        Validate configuration consistency.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Translation-only runs take a text file instead of audio/video input.
        if self.translate_text_input:
            if not self.translate_to:
                raise ValueError(
                    "'translate_text_input' requires 'translate_to' (target language)."
                )
            if not self.translate_from:
                raise ValueError(
                    "'translate_text_input' requires 'translate_from': the source "
                    "language of a plain text file cannot be auto-detected."
                )
        elif not self.input_file and not self.input_url:
            raise ValueError(
                "Either 'input_file', 'input_url', or 'translate_text_input' "
                "must be provided."
            )

        if self.translate_to is not None and not _is_iso639_1(self.translate_to):
            raise ValueError(
                f"Invalid 'translate_to' language code '{self.translate_to}'. "
                f"Expected an ISO 639-1 code such as 'en' or 'fr'."
            )
        if self.translate_from is not None and not _is_iso639_1(self.translate_from):
            raise ValueError(
                f"Invalid 'translate_from' language code '{self.translate_from}'. "
                f"Expected an ISO 639-1 code such as 'en' or 'fr'."
            )

        valid_backends = {"whisper_cpp", "faster_whisper", "openai", "mega_asr"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{self.backend}'. Valid options: {valid_backends}"
            )
        if self.fallback_backend and self.fallback_backend not in valid_backends:
            raise ValueError(
                f"Invalid fallback_backend '{self.fallback_backend}'. Valid options: {valid_backends}"
            )

        # Fail fast if mega_asr is asked to handle a language Qwen3-ASR cannot.
        if "mega_asr" in (self.backend, self.fallback_backend) and self.language:
            from transcriber.backends.mega_asr import MEGA_SUPPORTED_LANGUAGES
            if self.language not in MEGA_SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"Backend 'mega_asr' does not support language "
                    f"'{self.language}'. Supported languages: "
                    f"{sorted(MEGA_SUPPORTED_LANGUAGES)}."
                )

        valid_formats = {"txt", "json", "srt", "vtt"}
        for fmt in self.output_formats:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Invalid output format '{fmt}'. Valid options: {valid_formats}"
                )

        valid_wt_providers = {"native", "stable_ts", "whisperx"}
        if self.word_timestamps_provider not in valid_wt_providers:
            raise ValueError(
                f"Invalid word_timestamps_provider "
                f"'{self.word_timestamps_provider}'. Valid options: "
                f"{sorted(valid_wt_providers)}."
            )

        if self.workers < 1:
            raise ValueError("'workers' must be >= 1.")
        if self.chunk_duration_seconds < 10:
            raise ValueError("'chunk_duration_seconds' must be >= 10.")
        if self.max_retries < 0:
            raise ValueError("'max_retries' must be >= 0.")

        if self.backend == "openai" and not self.openai_api_key:
            env_key = os.environ.get("OPENAI_API_KEY")
            if env_key:
                self.openai_api_key = env_key
            else:
                raise ValueError(
                    "Backend 'openai' requires 'openai_api_key' in config or "
                    "OPENAI_API_KEY environment variable."
                )

        logger.debug("Configuration validated successfully.")

    @classmethod
    def from_dict(cls, data: dict, interpolate: bool = True) -> "TranscriptionConfig":
        """
        Build a TranscriptionConfig from a plain dictionary.

        interpolate expands ${ENV_VAR} placeholders in string values -- handy for
        config files, but pass interpolate=False for dicts built from untrusted
        input (e.g. the web UI) so caller-supplied values can't read the env.
        """
        source = _interpolate_dict(data) if interpolate else data
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in source.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str) -> "TranscriptionConfig":
        """
        Load configuration from a JSON file.

        Args:
            path: Path to the JSON config file.

        Returns:
            TranscriptionConfig instance populated from the file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.debug("Loaded config from file: %s", path)
        return cls.from_dict(data)

    def apply_overrides(self, overrides: dict) -> None:
        """
        Apply a dictionary of overrides onto this config instance.
        Only keys that correspond to existing fields and have non-None values are applied.

        Args:
            overrides: Dictionary of field_name -> value.
        """
        known_fields = {f.name for f in self.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        for key, value in overrides.items():
            if key in known_fields and value is not None:
                setattr(self, key, value)
                logger.debug("Config override applied: %s = %r", key, value)
