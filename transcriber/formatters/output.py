#!/usr/bin/env python3
# transcriber/formatters/output.py

"""
Output Formatter Module

Description:
Writes TranscriptionResult objects to disk in one or more formats:
  - txt  : plain text transcript
  - json : structured JSON with metadata and segments
  - srt  : SubRip subtitle format
  - vtt  : WebVTT subtitle format

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import json
import logging
import os
from pathlib import Path
from typing import List

from transcriber.backends.base import TranscriptionResult


logger = logging.getLogger(__name__)


def _seconds_to_srt_ts(seconds: float) -> str:
    """Convert a float seconds value to SRT timestamp: HH:MM:SS,mmm"""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    secs = total_s % 60
    total_m = total_s // 60
    mins = total_m % 60
    hours = total_m // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def _seconds_to_vtt_ts(seconds: float) -> str:
    """Convert a float seconds value to WebVTT timestamp: HH:MM:SS.mmm"""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    secs = total_s % 60
    total_m = total_s // 60
    mins = total_m % 60
    hours = total_m // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{ms:03d}"


class OutputFormatter:
    """
    Writes a TranscriptionResult to one or more output formats.

    All output files are placed in output_dir and named:
        {output_prefix}.{ext}
    """

    def __init__(self, output_dir: str = ".", output_prefix: str = "transcript") -> None:
        """
        Args:
            output_dir:    Directory where output files are written.
            output_prefix: Base filename prefix (without extension).
        """
        self.output_dir = output_dir
        self.output_prefix = output_prefix

    def write(
        self,
        result: TranscriptionResult,
        formats: List[str],
    ) -> List[str]:
        """
        Write the transcription result in all requested formats.

        Args:
            result:  The TranscriptionResult to format.
            formats: List of format strings: 'txt', 'json', 'srt', 'vtt'.

        Returns:
            List of absolute paths to the files that were written.

        Raises:
            ValueError: If an unsupported format is requested.
            OSError:    If a file cannot be written.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        written: List[str] = []
        dispatch = {
            "txt": self._write_txt,
            "json": self._write_json,
            "srt": self._write_srt,
            "vtt": self._write_vtt,
        }

        for fmt in formats:
            fmt = fmt.lower().strip()
            if fmt not in dispatch:
                raise ValueError(
                    f"Unsupported output format: '{fmt}'. "
                    f"Valid options: {sorted(dispatch.keys())}"
                )
            output_path = str(
                Path(self.output_dir) / f"{self.output_prefix}.{fmt}"
            )
            dispatch[fmt](result, output_path)
            written.append(output_path)
            logger.info("Written %s output: %s", fmt.upper(), output_path)

        return written

    # ------------------------------------------------------------------
    # Format writers
    # ------------------------------------------------------------------

    def _write_txt(self, result: TranscriptionResult, path: str) -> None:
        """Write plain text transcript."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(result.text)
            fh.write("\n")

    def _write_json(self, result: TranscriptionResult, path: str) -> None:
        """Write structured JSON output including metadata and segments."""
        data = {
            "text": result.text,
            "language": result.language,
            "backend": result.backend_name,
            "duration": result.duration,
            "source_file": result.source_file,
            "translated": result.translated,
            "source_language": result.source_language,
            "segments": result.segments,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    def _write_srt(self, result: TranscriptionResult, path: str) -> None:
        """
        Write SubRip (.srt) subtitle file.

        If no segments with timestamps are available, writes the entire
        transcript as a single subtitle entry with dummy timestamps.
        """
        segments = result.segments

        if not segments:
            # Single-block fallback
            segments = [{"start": 0.0, "end": 0.0, "text": result.text}]

        with open(path, "w", encoding="utf-8") as fh:
            for idx, seg in enumerate(segments, start=1):
                start = _seconds_to_srt_ts(float(seg.get("start", 0.0)))
                end = _seconds_to_srt_ts(float(seg.get("end", 0.0)))
                text = str(seg.get("text", "")).strip()

                fh.write(f"{idx}\n")
                fh.write(f"{start} --> {end}\n")
                fh.write(f"{text}\n")
                fh.write("\n")

    def _write_vtt(self, result: TranscriptionResult, path: str) -> None:
        """
        Write WebVTT (.vtt) subtitle file.

        If no segments with timestamps are available, writes the entire
        transcript as a single cue with dummy timestamps.
        """
        segments = result.segments

        if not segments:
            segments = [{"start": 0.0, "end": 0.0, "text": result.text}]

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("WEBVTT\n\n")
            for idx, seg in enumerate(segments, start=1):
                start = _seconds_to_vtt_ts(float(seg.get("start", 0.0)))
                end = _seconds_to_vtt_ts(float(seg.get("end", 0.0)))
                text = str(seg.get("text", "")).strip()

                fh.write(f"{idx}\n")
                fh.write(f"{start} --> {end}\n")
                fh.write(f"{text}\n")
                fh.write("\n")
