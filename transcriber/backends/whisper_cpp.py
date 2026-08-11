#!/usr/bin/env python3
# transcriber/backends/whisper_cpp.py

"""
Whisper.cpp Backend

Description:
Invokes a compiled whisper.cpp binary via subprocess to perform transcription.
Parses the plain-text output produced by the binary.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

from transcriber.backends.base import TranscriptionBackend, TranscriptionResult


logger = logging.getLogger(__name__)

# Regex for whisper.cpp timestamp lines: [HH:MM:SS.mmm --> HH:MM:SS.mmm]  text
_TIMESTAMP_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)"
)


def _ts_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS.mmm or HH:MM:SS,mmm timestamp string to float seconds."""
    parts = ts.split(":")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2].replace(",", "."))
    return hours * 3600 + minutes * 60 + seconds


class WhisperCppBackend(TranscriptionBackend):
    """
    Transcription backend that calls a compiled whisper.cpp binary.

    The binary is expected to accept:
        whisper -m <model> -f <audio_file> [--language <lang>] [--output-json] ...
    """

    def __init__(
        self,
        binary_path: str = "whisper",
        model_path: str = "models/ggml-base.bin",
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            binary_path: Path to (or name of) the whisper.cpp binary.
            model_path:  Path to the GGML model file.
            extra_args:  Additional CLI arguments forwarded verbatim to the binary.
        """
        self.binary_path = binary_path
        self.model_path = model_path
        self.extra_args: List[str] = extra_args or []

    @property
    def name(self) -> str:
        return "whisper_cpp"

    def is_available(self) -> bool:
        """Return True if the whisper.cpp binary can be found on PATH or at the given path."""
        found = shutil.which(self.binary_path) is not None or Path(self.binary_path).is_file()
        if not found:
            logger.debug("whisper.cpp binary not found: %s", self.binary_path)
        return found

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Run whisper.cpp on the given audio file.

        Args:
            audio_path: Path to the audio file (WAV preferred for whisper.cpp).
            language:   ISO 639-1 language code, or None for auto-detect.

        Returns:
            TranscriptionResult with text and segments parsed from output.

        Raises:
            RuntimeError: If the subprocess exits with a non-zero return code.
        """
        if not Path(audio_path).is_file():
            raise RuntimeError(f"Audio file does not exist: {audio_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_base = str(Path(tmp_dir) / "out")

            want_words = bool(kwargs.get("word_timestamps"))

            cmd: List[str] = [
                self.binary_path,
                "-m", self.model_path,
                "-f", audio_path,
                # --output-json-full adds per-token timestamps, which we group
                # into words; plain --output-json otherwise.
                "--output-json-full" if want_words else "--output-json",
                "-of", output_base,
            ]

            if language:
                cmd += ["--language", language]

            cmd += self.extra_args

            logger.debug("whisper.cpp command: %s", " ".join(cmd))

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"whisper.cpp timed out processing: {audio_path}"
                ) from exc
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"whisper.cpp binary not found: {self.binary_path}"
                ) from exc

            if proc.returncode != 0:
                stderr_snippet = proc.stderr[-500:] if proc.stderr else "(no stderr)"
                raise RuntimeError(
                    f"whisper.cpp exited with code {proc.returncode}.\n"
                    f"stderr: {stderr_snippet}"
                )

            # Try to parse the JSON output file whisper.cpp produces
            json_output_path = Path(output_base + ".json")
            if json_output_path.is_file():
                return self._parse_json_output(
                    json_output_path, audio_path, want_words=want_words
                )

            # Fallback: parse stdout for timestamp lines
            return self._parse_text_output(proc.stdout, audio_path)

    @staticmethod
    def _tokens_to_words(tokens: List[dict]) -> List[dict]:
        """
        Group whisper.cpp per-token entries (from --output-json-full) into words.

        A whisper token that begins with a space starts a new word; special
        tokens (e.g. '[_BEG_]', '[_TT_..]') are skipped. Each word's start/end
        come from its first/last token timestamps.
        """
        words: List[dict] = []
        current = None  # {"word": str, "start": float, "end": float}

        for tok in tokens:
            text = tok.get("text", "")
            if not text or (text.startswith("[_") and text.endswith("]")):
                continue
            offsets = tok.get("offsets") or {}
            # offsets are in milliseconds; fall back to the timestamps strings
            if "from" in offsets and "to" in offsets:
                t0 = offsets["from"] / 1000.0
                t1 = offsets["to"] / 1000.0
            else:
                ts = tok.get("timestamps") or {}
                t0 = _ts_to_seconds(ts.get("from", "00:00:00.000"))
                t1 = _ts_to_seconds(ts.get("to", "00:00:00.000"))

            starts_word = text.startswith(" ") or current is None
            if starts_word:
                if current is not None:
                    words.append(current)
                current = {"word": text.strip(), "start": t0, "end": t1}
            else:
                current["word"] += text
                current["end"] = t1

        if current is not None and current["word"]:
            words.append(current)
        return words

    def _parse_json_output(
        self, json_path: Path, source_file: str, want_words: bool = False
    ) -> TranscriptionResult:
        """Parse whisper.cpp's JSON output file."""
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        segments = []
        texts = []
        detected_language = None

        # whisper.cpp JSON schema: {"transcription": [{"timestamps": {...}, "text": "..."}]}
        # With --output-json-full each item also carries a "tokens" list.
        for item in data.get("transcription", []):
            text = item.get("text", "").strip()
            ts = item.get("timestamps", {})
            start_str = ts.get("from", "00:00:00.000")
            end_str = ts.get("to", "00:00:00.000")
            entry = {
                "start": _ts_to_seconds(start_str),
                "end": _ts_to_seconds(end_str),
                "text": text,
            }
            if want_words and item.get("tokens"):
                seg_words = self._tokens_to_words(item["tokens"])
                if seg_words:
                    entry["words"] = seg_words
            segments.append(entry)
            texts.append(text)

        # Some versions expose the detected language at top level
        detected_language = data.get("language") or data.get("params", {}).get("language")

        return TranscriptionResult(
            text=" ".join(texts),
            language=detected_language,
            segments=segments,
            source_file=source_file,
            backend_name=self.name,
        )

    def _parse_text_output(
        self, stdout: str, source_file: str
    ) -> TranscriptionResult:
        """
        Parse whisper.cpp text output as a fallback when JSON file is unavailable.
        Handles both timestamped lines and plain text output.
        """
        segments = []
        texts = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            match = _TIMESTAMP_RE.match(line)
            if match:
                start_str, end_str, text = match.groups()
                text = text.strip()
                if text:
                    segments.append({
                        "start": _ts_to_seconds(start_str),
                        "end": _ts_to_seconds(end_str),
                        "text": text,
                    })
                    texts.append(text)
            else:
                # Plain text line without timestamp
                texts.append(line)

        full_text = " ".join(texts)
        return TranscriptionResult(
            text=full_text,
            segments=segments,
            source_file=source_file,
            backend_name=self.name,
        )
