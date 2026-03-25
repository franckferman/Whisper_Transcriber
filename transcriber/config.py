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

    # Processing
    language: Optional[str] = None
    chunk_duration_seconds: int = 600
    workers: int = 2
    temp_dir: Optional[str] = None

    # Output
    output_formats: List[str] = field(default_factory=lambda: ["txt"])
    output_dir: str = "."
    output_prefix: str = "transcript"

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
        if not self.input_file and not self.input_url:
            raise ValueError("Either 'input_file' or 'input_url' must be provided.")

        valid_backends = {"whisper_cpp", "faster_whisper", "openai"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{self.backend}'. Valid options: {valid_backends}"
            )
        if self.fallback_backend and self.fallback_backend not in valid_backends:
            raise ValueError(
                f"Invalid fallback_backend '{self.fallback_backend}'. Valid options: {valid_backends}"
            )

        valid_formats = {"txt", "json", "srt", "vtt"}
        for fmt in self.output_formats:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Invalid output format '{fmt}'. Valid options: {valid_formats}"
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
    def from_dict(cls, data: dict) -> "TranscriptionConfig":
        """Build a TranscriptionConfig from a plain dictionary."""
        interpolated = _interpolate_dict(data)
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in interpolated.items() if k in known_fields}
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
