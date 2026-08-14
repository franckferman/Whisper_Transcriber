#!/usr/bin/env python3
# main.py

"""
whispr - CLI Entry Point

Description:
Command-line interface for the modular transcription system.
Supports local files, YouTube URLs, and generic HTTP URLs.
Multiple backends: whisper.cpp, faster-whisper, OpenAI API.

Created By  : Franck FERMAN
Version     : 2.0.0

Usage examples:
    python main.py --config config.json
    python main.py --url https://youtube.com/watch?v=... --backend whisper_cpp --workers 4 --language fr
    python main.py --file video.mp4 --backend faster_whisper --workers 2
    python main.py --file audio.wav --backend openai
    python main.py --dry-run --url https://... --backend whisper_cpp
"""

import argparse
import sys

from transcriber.config import TranscriptionConfig
from transcriber.logger import setup_logging
from transcriber.managers.transcription import TranscriptionManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Modular video/audio transcription system with multi-backend support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  Load full config from file:
    python main.py --config config.json

  Transcribe a YouTube video with whisper.cpp, 4 workers, French:
    python main.py --url https://youtube.com/watch?v=... --backend whisper_cpp --workers 4 --language fr

  Transcribe a local video with faster-whisper, 2 parallel workers:
    python main.py --file video.mp4 --backend faster_whisper --workers 2

  Transcribe a local audio file via OpenAI API:
    python main.py --file audio.wav --backend openai --openai-key sk-...

  Dry run (no output written):
    python main.py --dry-run --file audio.mp3 --backend whisper_cpp

  Transcribe then translate locally to French (keeps the original):
    python main.py --file talk.mp4 --language en --translate-to fr

  Translate an existing text file, no transcription:
    python main.py --translate-text notes.txt --translate-from en --translate-to fr
        """,
    )

    # ---- Input ----
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Path to a local audio/video file.",
    )
    input_group.add_argument(
        "--url", "-u",
        metavar="URL",
        help="URL to a YouTube video or direct audio/video URL.",
    )
    input_group.add_argument(
        "--translate-text",
        metavar="PATH",
        help=(
            "Translate an existing text file locally, with no transcription. "
            "Requires --translate-to and --translate-from."
        ),
    )

    parser.add_argument(
        "--config", "-c",
        metavar="FILE",
        help="Path to a JSON configuration file. CLI arguments override config file values.",
    )

    backend = parser.add_argument_group("backend")
    backend.add_argument(
        "--backend", "-b",
        choices=["whisper_cpp", "faster_whisper", "openai", "mega_asr"],
        metavar="BACKEND",
        help="Transcription backend: whisper_cpp | faster_whisper | openai | mega_asr (default: faster_whisper).",
    )
    backend.add_argument(
        "--fallback-backend",
        choices=["whisper_cpp", "faster_whisper", "openai", "mega_asr"],
        metavar="BACKEND",
        help="Fallback backend if the primary backend fails all retries.",
    )
    backend.add_argument(
        "--whisper-binary",
        metavar="PATH",
        help="Path to the whisper.cpp binary (default: 'whisper').",
    )
    backend.add_argument(
        "--whisper-model",
        metavar="PATH",
        help="Path to the GGML model file for whisper.cpp.",
    )
    backend.add_argument(
        "--fw-model",
        metavar="NAME",
        help="faster-whisper model size (tiny/base/small/medium/large-v2, default: base).",
    )
    backend.add_argument(
        "--fw-device",
        choices=["cpu", "cuda"],
        metavar="DEVICE",
        help="faster-whisper inference device (cpu or cuda, default: cpu).",
    )
    backend.add_argument(
        "--openai-key",
        metavar="KEY",
        help="OpenAI API key (can also be set via OPENAI_API_KEY env var).",
    )
    backend.add_argument(
        "--openai-model",
        metavar="MODEL",
        help="OpenAI model name (default: whisper-1).",
    )
    backend.add_argument(
        "--mega-repo",
        metavar="PATH",
        help="Path to a local xzf-thu/Mega-ASR clone (provides the MegaASR wrapper).",
    )
    backend.add_argument(
        "--mega-ckpt",
        metavar="PATH",
        help="Mega-ASR checkpoint root (default: <mega-repo>/ckpt/Mega-ASR).",
    )
    backend.add_argument(
        "--mega-device",
        metavar="MAP",
        help="Mega-ASR device_map: 'cuda:0' | 'mps' | 'cpu' (default: auto).",
    )
    backend.add_argument(
        "--mega-allow-cpu",
        action="store_true",
        help="Allow Mega-ASR to run on CPU (slow for a 1.7B model; off by default).",
    )
    backend.add_argument(
        "--mega-force-lora",
        action="store_true",
        help="Mount the Mega-ASR LoRA regardless of language (en/zh-tuned only).",
    )

    processing = parser.add_argument_group("processing")
    processing.add_argument(
        "--language", "-l",
        metavar="LANG",
        help="ISO 639-1 language code (e.g. 'fr', 'en'). Auto-detect if omitted.",
    )
    processing.add_argument(
        "--chunk-duration",
        type=int,
        metavar="SECONDS",
        help="Duration of each audio chunk in seconds (default: 600).",
    )
    processing.add_argument(
        "--workers", "-w",
        type=int,
        metavar="N",
        help="Number of parallel transcription workers (default: 2).",
    )
    processing.add_argument(
        "--temp-dir",
        metavar="DIR",
        help="Directory for temporary files (default: OS temp dir).",
    )

    translation = parser.add_argument_group(
        "translation (local, optional 'argostranslate' extra)")
    translation.add_argument(
        "--translate-to",
        metavar="LANG",
        help=(
            "Target ISO 639-1 code (e.g. 'en', 'fr'). Enables local translation "
            "of the transcript after transcription; the original is kept and a "
            "'{prefix}.{LANG}.{fmt}' copy is written."
        ),
    )
    translation.add_argument(
        "--translate-from",
        metavar="LANG",
        help=(
            "Source ISO 639-1 code for translation. Defaults to the detected/"
            "--language value; required with --translate-text."
        ),
    )
    translation.add_argument(
        "--translate-model",
        metavar="PATH",
        help="Path to a local .argosmodel package for fully offline translation.",
    )
    translation.add_argument(
        "--no-translate-download",
        action="store_true",
        help="Never download translation models; use only locally installed ones.",
    )

    words = parser.add_argument_group("word-level timestamps (opt-in, additive)")
    words.add_argument(
        "--word-timestamps",
        action="store_true",
        help=(
            "Emit per-word timestamps inside each segment (JSON). Off by default; "
            "adds a 'words' list per segment without changing existing output."
        ),
    )
    words.add_argument(
        "--word-timestamps-provider",
        choices=["native", "stable_ts", "whisperx"],
        metavar="PROVIDER",
        help=(
            "Where word timing comes from: native (backend's own, default), "
            "stable_ts or whisperx (post-hoc forced alignment, optional extras)."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--format", "-F",
        dest="output_format",
        metavar="FMT",
        help=(
            "Output format(s), comma-separated: txt,json,srt,vtt "
            "(default: txt). Example: --format txt,srt"
        ),
    )
    output.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        help="Directory where output files are written (default: current directory).",
    )
    output.add_argument(
        "--output-prefix",
        metavar="PREFIX",
        help="Base filename prefix for output files (default: transcript).",
    )
    output.add_argument(
        "--max-retries",
        type=int,
        metavar="N",
        help="Maximum retry attempts per chunk on backend failure (default: 3).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without performing transcription or writing files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        help="Write logs to this file. Use 'auto' for a timestamped filename.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # ---- Logging setup (before any other output) ----
    setup_logging(
        debug=args.debug,
        log_file=args.log_file,
    )

    # ---- Build config ----
    if args.config:
        config = TranscriptionConfig.from_json_file(args.config)
    else:
        config = TranscriptionConfig()

    # ---- Parse output formats from CLI ----
    output_formats = None
    if args.output_format:
        output_formats = [f.strip() for f in args.output_format.split(",") if f.strip()]

    # ---- Build overrides dict from CLI args ----
    overrides = {}
    if args.file:
        overrides["input_file"] = args.file
    if args.url:
        overrides["input_url"] = args.url
    if args.translate_text:
        overrides["translate_text_input"] = args.translate_text
    if args.translate_to:
        overrides["translate_to"] = args.translate_to
    if args.translate_from:
        overrides["translate_from"] = args.translate_from
    if args.translate_model:
        overrides["translate_package_path"] = args.translate_model
    if args.no_translate_download:
        overrides["translate_allow_download"] = False
    if args.backend:
        overrides["backend"] = args.backend
    if args.fallback_backend:
        overrides["fallback_backend"] = args.fallback_backend
    if args.language:
        overrides["language"] = args.language
    if args.chunk_duration is not None:
        overrides["chunk_duration_seconds"] = args.chunk_duration
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.word_timestamps:
        overrides["word_timestamps"] = True
    if args.word_timestamps_provider:
        overrides["word_timestamps_provider"] = args.word_timestamps_provider
    if args.temp_dir:
        overrides["temp_dir"] = args.temp_dir
    if output_formats:
        overrides["output_formats"] = output_formats
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.output_prefix:
        overrides["output_prefix"] = args.output_prefix
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries
    if args.dry_run:
        overrides["dry_run"] = True
    if args.debug:
        overrides["debug"] = True
    if args.log_file:
        overrides["log_file"] = args.log_file

    # Backend-specific
    if args.whisper_binary:
        overrides["whisper_cpp_binary"] = args.whisper_binary
    if args.whisper_model:
        overrides["whisper_cpp_model"] = args.whisper_model
    if args.fw_model:
        overrides["faster_whisper_model"] = args.fw_model
    if args.fw_device:
        overrides["faster_whisper_device"] = args.fw_device
    if args.openai_key:
        overrides["openai_api_key"] = args.openai_key
    if args.openai_model:
        overrides["openai_model"] = args.openai_model
    if args.mega_repo:
        overrides["mega_asr_repo_dir"] = args.mega_repo
    if args.mega_ckpt:
        overrides["mega_asr_ckpt_dir"] = args.mega_ckpt
    if args.mega_device:
        overrides["mega_asr_device_map"] = args.mega_device
    if args.mega_allow_cpu:
        overrides["mega_asr_allow_cpu"] = True
    if args.mega_force_lora:
        overrides["mega_asr_force_lora"] = True

    config.apply_overrides(overrides)

    # ---- Run ----
    try:
        manager = TranscriptionManager(config)
        manager.run()
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
