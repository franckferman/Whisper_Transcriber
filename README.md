<div align="center">

# whispr

**Turn audio and video into text, subtitles, or translations with OpenAI's Whisper.**

**Runs fully local or in the cloud, from the command line or your browser.**

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Backends](https://img.shields.io/badge/backends-whisper.cpp_%7C_faster--whisper_%7C_openai_%7C_mega--asr-555555?style=flat-square)](https://github.com/franckferman/whispr)

</div>

---

## Overview

whispr transcribes audio and video with OpenAI's Whisper, from a scriptable CLI or a browser UI. Choose the engine that fits the job: three interchangeable general-purpose backends, local or cloud, plus an optional Mega-ASR specialist for degraded audio. It returns plain text, subtitles (SRT/VTT), or JSON, and can translate the result locally, fully offline.

Two interfaces ship with the same codebase:

- A **CLI** (`main.py`) for scripting, automation, and direct use.
- A **web UI** (`web/`) for browser-based use: upload a file or paste a URL, pick a backend, download the result.

**Key capabilities:**

| Area | Details |
|---|---|
| **Backends** | whisper.cpp (GGML subprocess), faster-whisper (CTranslate2), OpenAI Whisper API, and an optional Mega-ASR backend for degraded audio |
| **Input** | Local files, YouTube URLs (yt-dlp), direct HTTP URLs |
| **Input formats** | MP3, MP4, WAV, M4A, MKV, WEBM, OGG, FLAC, AVI, MOV, and any format supported by ffmpeg |
| **Output formats** | Plain text, JSON (with segments and timestamps), SRT subtitles, WebVTT subtitles |
| **Chunking** | ffmpeg-based splitting for arbitrarily long files, transparent to the user |
| **Configuration** | JSON file + CLI overrides + `${ENV_VAR}` interpolation |
| **Web interface** | FastAPI + vanilla JS, real-time WebSocket log, per-backend options |

---

## Installation

**System prerequisites:** Python 3.9+ and ffmpeg (`install.sh` installs ffmpeg for you if it is missing).

### Interactive installer (recommended)

```bash
git clone https://github.com/franckferman/whispr.git
cd whispr
bash install.sh
```

The installer asks two questions:

**1. Interface:**
```
[1] CLI only
[2] Web UI  (adds fastapi, uvicorn, python-multipart)
[3] Both
```

**2. Backend:**
```
[1] faster-whisper  (local, no API key, recommended)
[2] OpenAI API      (cloud, requires OPENAI_API_KEY)
[3] whisper.cpp     (local C++ binary, requires compilation)
[4] All
[5] Skip
```

It verifies Python and ffmpeg, creates a `.venv/`, installs the core deps plus the interface and backend(s) you pick, copies `config.example.json` to `config.json`, and (for whisper.cpp) clones and compiles the binary and records its paths in the config.

After installation:
```bash
source .venv/bin/activate
```

### Manual installation

```bash
git clone https://github.com/franckferman/whispr.git
cd whispr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# faster-whisper
pip install faster-whisper

# OpenAI
pip install openai

# Web UI
pip install fastapi "uvicorn[standard]" python-multipart
```

### whisper.cpp (manual build)

```bash
make whisper-cpp-setup                        # clone, compile, download base model
make whisper-cpp-setup WHISPER_MODEL=small    # or small / medium / large-v2
```

After compilation the binary is at `./whisper.cpp/build/bin/whisper-cli`. Set this path in `config.json` or pass it via `--whisper-binary`.

### Verify

```bash
make check
```

Validates all Python imports and verifies the CLI entry point responds correctly.

---

## Makefile Reference

```bash
make help                 # List all targets and variables
make install              # Full setup via install.sh
make venv                 # Create the virtualenv only
make deps                 # Core Python deps only
make deps-faster-whisper  # + faster-whisper
make deps-openai          # + openai
make deps-all             # Core + faster-whisper + openai (not the heavy extras)
make deps-translate       # + local translation (Argos; heavy, pulls torch)
make deps-align           # + premium word-timestamp providers (stable-ts / whisperx)
make deps-mega            # + Mega-ASR backend (heavy, GPU-oriented)

make whisper-cpp-build    # Clone and compile whisper.cpp
make whisper-cpp-model    # Download model (WHISPER_MODEL=base|tiny|small|medium|large-v2)
make whisper-cpp-setup    # Build + model in one step

make run FILE=video.mp4                         # Transcribe (faster_whisper, fr by default)
make run URL=https://...                        # Transcribe a URL
make run FILE=video.mp4 BACKEND=whisper_cpp     # Specific backend
make run FILE=video.mp4 LANGUAGE=en WORKERS=4   # Override defaults
make run-whisper-cpp FILE=video.mp4
make run-faster-whisper FILE=video.mp4
make run-openai FILE=video.mp4
make run-mega FILE=audio.wav MEGA_REPO=/path/to/Mega-ASR   # add MEGA_DEVICE=cuda:0

make run-srt FILE=video.mp4                     # SRT output only
make run-all-formats FILE=video.mp4             # txt + json + srt + vtt
make run-words FILE=talk.mp4                    # Per-word timestamps (PROVIDER=native|stable_ts|whisperx)

make run-translate FILE=talk.mp4 TO=fr          # Transcribe, then translate to TO=
make translate-text IN=notes.txt FROM=en TO=fr  # Translate an existing text file

make dry-run FILE=video.mp4                     # Preview without executing
make run-config                                 # Use config.json

make web                                        # Start web UI at http://localhost:8000
make web-install                                # Install web deps only

make check                                      # Validate imports + CLI
make clean                                      # Remove output files and __pycache__
make clean-all                                  # Remove venv too
```

---

## CLI Usage

```
python main.py [--file PATH | --url URL] [options]
```

If `--backend` is not specified, the value from `config.json` is used (default: `faster_whisper`). The CLI does not perform automatic backend detection: set your preferred backend in `config.json` once and omit the flag on subsequent runs.

### Examples

```bash
source .venv/bin/activate

# Local file, auto-detect language
python main.py --file video.mp4 --backend faster_whisper

# Explicit language (skips the 30 s language detection pass)
python main.py --file video.mp4 --backend faster_whisper --language fr

# YouTube URL
python main.py --url "https://youtube.com/watch?v=..." --backend faster_whisper --language fr

# Multiple output formats
python main.py --file video.mp4 --backend faster_whisper --format txt,srt,vtt,json

# whisper.cpp
python main.py --file video.mp4 --backend whisper_cpp \
    --whisper-binary ./whisper.cpp/build/bin/whisper-cli \
    --whisper-model  ./whisper.cpp/models/ggml-base.bin \
    --language fr

# OpenAI API
python main.py --file audio.mp3 --backend openai --openai-key sk-...

# Primary + fallback: re-run with OpenAI if faster-whisper exhausts retries
python main.py --file video.mp4 --backend faster_whisper --fallback-backend openai

# Long file: 5-minute chunks, 4 workers
python main.py --file film.mp4 --backend faster_whisper \
    --chunk-duration 300 --workers 4 --language fr --format txt,srt

# Dry run
python main.py --dry-run --file video.mp4 --backend faster_whisper

# Transcribe, then translate the result to French (keeps the original)
python main.py --file talk.mp4 --language en --translate-to fr --format txt,srt

# Translate an existing transcript, no transcription
python main.py --translate-text transcript.txt --translate-from en --translate-to fr

# Load config from file
python main.py --config config.json
```

> **YouTube and yt-dlp.** YouTube now deciphers its stream URLs with a JavaScript
> challenge, and yt-dlp needs a JS runtime to solve it. Without one, yt-dlp falls
> back to player clients (android/tv) that skip the challenge, so short videos still
> download, but some formats are missing, speeds can be throttled, and the path is
> deprecated. For reliable YouTube support, install **Deno** on the host (yt-dlp
> enables it by default and runs the untrusted player JS in its sandbox):
>
> ```bash
> emerge dev-lang/deno-bin      # Gentoo; elsewhere see https://deno.land
> ```
>
> yt-dlp auto-detects it, no whispr config needed. This is a host dependency; it
> only affects YouTube URLs, not file uploads or direct HTTP links.

### All flags

| Flag | Default | Description |
|---|---|---|
| `--file`, `-f` | | Local audio/video file |
| `--url`, `-u` | | YouTube URL or direct HTTP link |
| `--config`, `-c` | | JSON config file (CLI flags override) |
| `--backend`, `-b` | `faster_whisper` | `whisper_cpp` \| `faster_whisper` \| `openai` \| `mega_asr` |
| `--fallback-backend` | | Secondary backend if primary fails all retries |
| `--whisper-binary` | `whisper` | Path to the whisper.cpp binary |
| `--whisper-model` | | Path to the GGML `.bin` model file |
| `--fw-model` | `base` | faster-whisper model size |
| `--fw-device` | `cpu` | faster-whisper device: `cpu` \| `cuda` |
| `--openai-key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--openai-model` | `whisper-1` | OpenAI model identifier |
| `--mega-repo` | | Path to a local xzf-thu/Mega-ASR clone |
| `--mega-ckpt` | | Mega-ASR checkpoint root (default: `<repo>/ckpt/Mega-ASR`) |
| `--mega-device` | auto | Mega-ASR device map: `cuda:0`, `mps`, `cpu` |
| `--mega-allow-cpu` | | Allow Mega-ASR on CPU (slow for a 1.7B model) |
| `--mega-force-lora` | | Mount the LoRA regardless of language (en/zh-tuned only) |
| `--language`, `-l` | auto-detect | ISO 639-1 code: `fr`, `en`, `es`, ... |
| `--chunk-duration` | `600` | Chunk size in seconds |
| `--workers`, `-w` | `2` | Parallel transcription threads |
| `--temp-dir` | | Directory for temporary files (OS default if unset) |
| `--word-timestamps` | | Emit per-word timestamps inside each segment (JSON) |
| `--word-timestamps-provider` | `native` | Word timing source: `native`, `stable_ts`, `whisperx` |
| `--format`, `-F` | `txt` | Output formats, comma-separated: `txt,json,srt,vtt` |
| `--output-dir`, `-o` | `.` | Output directory |
| `--output-prefix` | `transcript` | Output filename prefix |
| `--translate-to` | | Target ISO 639-1 code; translate the transcript locally |
| `--translate-from` | detected | Source ISO 639-1 code; required with `--translate-text` |
| `--translate-text` | | Translate an existing text file, no transcription |
| `--translate-model` | | Local `.argosmodel` package for offline translation |
| `--no-translate-download` | | Never download models; use only installed ones |
| `--max-retries` | `3` | Retry attempts per chunk on failure |
| `--dry-run` | | Print plan without executing |
| `--debug` | | Enable DEBUG logging |
| `--log-file` | | Write logs to file (`auto` for timestamped name) |

### faster-whisper model sizes

Models are downloaded on first use and cached at `~/.cache/huggingface/hub/`.

| Model | Size | Speed (CPU) | Quality | Use case |
|---|---|---|---|---|
| `tiny` | ~75 MB | Fastest | Basic | Quick drafts, noisy audio |
| `base` | ~150 MB | Fast | Good | Default, everyday use |
| `small` | ~500 MB | Moderate | Better | Interviews, mixed accents |
| `medium` | ~1.5 GB | Slow | High | Difficult content |
| `large-v2` | ~3 GB | Slowest | Best | Maximum accuracy |

### GPU acceleration

```bash
python main.py --file video.mp4 --backend faster_whisper --fw-device cuda
```

In `config.json`:
```json
"faster_whisper_device": "cuda",
"faster_whisper_compute_type": "float16"
```

`float16` on GPU, `int8` on CPU.

### Output formats

All four formats are produced by the same formatter, independently of the backend.

| Format | Description |
|---|---|
| `txt` | Plain text, no timestamps |
| `json` | Full output: segments, timestamps, language, metadata |
| `srt` | SubRip subtitles (VLC, ffmpeg, Premiere) |
| `vtt` | WebVTT for the HTML5 `<video>` element |

### Chunking

Long files are split into fixed-duration chunks before transcription. Each chunk is processed independently and results are merged with corrected absolute timestamps.

The default is 600 seconds (10 minutes). Reduce for very long files to lower peak memory, increase to reduce per-chunk overhead:

```bash
python main.py --file film.mp4 --chunk-duration 300 --workers 4
```

The OpenAI API's 25 MB limit per request is handled transparently by the chunking layer.

### Retry and fallback

Each chunk is retried independently with exponential backoff:

```
attempt 1 -> wait 1 s -> attempt 2 -> wait 2 s -> attempt 3 -> ...
```

If all retries fail and a fallback backend is configured, the entire job restarts from scratch on the fallback:

```bash
python main.py --file video.mp4 --backend faster_whisper \
    --fallback-backend openai --max-retries 5
```

---

## Configuration

Settings are read from `config.json` (if present). Any value can be overridden with a CLI flag. CLI flags always take precedence over the file.

```bash
cp config.example.json config.json
```

String values support `${ENV_VAR}` interpolation:
```json
"openai_api_key": "${OPENAI_API_KEY}"
```

### Full reference

| Field | Default | Description |
|---|---|---|
| `backend` | `faster_whisper` | Primary backend: `faster_whisper`, `whisper_cpp`, `openai`, `mega_asr` |
| `fallback_backend` | `null` | Secondary backend used if primary exhausts all retries |
| `faster_whisper_model` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v2` |
| `faster_whisper_device` | `cpu` | Compute device: `cpu` or `cuda` |
| `faster_whisper_compute_type` | `int8` | Quantization: `int8`, `float16`, `float32` |
| `whisper_cpp_binary` | `whisper` | Path to the compiled whisper.cpp binary |
| `whisper_cpp_model` | `models/ggml-base.bin` | Path to the GGML `.bin` model file |
| `whisper_cpp_extra_args` | `[]` | Additional CLI arguments forwarded verbatim to the binary |
| `openai_api_key` | `${OPENAI_API_KEY}` | OpenAI API key |
| `openai_model` | `whisper-1` | OpenAI model identifier |
| `mega_asr_repo_dir` | `null` | Path to a local xzf-thu/Mega-ASR clone (provides the wrapper) |
| `mega_asr_ckpt_dir` | `null` | Checkpoint root (default: `<repo>/ckpt/Mega-ASR`) |
| `mega_asr_device_map` | `null` | Device map: `cuda:0`, `mps`, `cpu` |
| `mega_asr_allow_cpu` | `false` | Allow CPU inference (slow for a 1.7B model) |
| `mega_asr_force_lora` | `false` | Mount the LoRA regardless of language (en/zh-tuned only) |
| `language` | `null` | ISO 639-1 code (`fr`, `en`, ...) or `null` for auto-detect |
| `chunk_duration_seconds` | `600` | Duration of each audio chunk in seconds |
| `workers` | `2` | Number of parallel transcription threads |
| `temp_dir` | `null` | Temporary directory (system default if null) |
| `word_timestamps` | `false` | Emit per-word timestamps inside each segment (JSON) |
| `word_timestamps_provider` | `native` | Word timing source: `native`, `stable_ts`, `whisperx` |
| `output_formats` | `["txt"]` | Output formats: `txt`, `json`, `srt`, `vtt` |
| `output_dir` | `.` | Directory where output files are written |
| `output_prefix` | `transcript` | Filename prefix for output files |
| `translate_to` | `null` | Target ISO 639-1 code; set it to translate the transcript locally |
| `translate_from` | `null` | Source override; defaults to the detected/`language` value |
| `translate_package_path` | `null` | Local `.argosmodel` file for fully offline translation |
| `translate_allow_download` | `true` | Allow one-time model download from the Argos index |
| `translate_text_input` | `null` | Translate an existing text file instead of transcribing audio |
| `max_retries` | `3` | Retry attempts per chunk before declaring failure |
| `retry_base_delay` | `1.0` | Base delay in seconds for exponential backoff |
| `retry_max_delay` | `30.0` | Maximum backoff delay cap in seconds |
| `dry_run` | `false` | Print the execution plan without running anything |
| `debug` | `false` | Enable DEBUG-level logging |
| `log_file` | `null` | Write logs to a file (`auto` generates a timestamped filename) |
| `ytdlp_format` | `bestaudio/best` | yt-dlp format selector |
| `ytdlp_output_template` | `%(title)s.%(ext)s` | yt-dlp output filename template |

---

## Backends

The three general-purpose backends execute Whisper inference through different runtimes, model formats, and compute environments. Transcription quality is identical for the same model size.

### whisper.cpp (`whisper_cpp`)

whisper.cpp is a standalone C++ reimplementation of Whisper by [Georgi Gerganov](https://github.com/ggerganov/whisper.cpp), built on [GGML](https://github.com/ggerganov/ggml), a minimal C tensor library with no dependencies beyond the C standard library.

**Model format:**
GGML stores weights as flat tensors in a custom binary `.bin` format, with a header describing the architecture. Quantization is applied statically at conversion time:

| Quantization | Bits/weight | Notes |
|---|---|---|
| `f32` | 32 | Reference, no compression |
| `f16` | 16 | Default for distributed `.bin` models |
| `q8_0` | 8 | 8-bit symmetric per-block |
| `q5_0` / `q5_1` | 5 | 5-bit quantization |
| `q4_0` / `q4_1` | 4 | Lowest memory footprint |

Quantized variants can be produced locally with the `quantize` tool bundled with whisper.cpp.

**CPU optimizations:**
GGML selects SIMD intrinsics at compile time: **AVX**, **AVX2**, **AVX-512** on x86-64, **NEON** on ARM (Apple Silicon, Raspberry Pi). This gives near-metal CPU throughput without any GPU requirement.

**Input:**
whisper.cpp requires **16 kHz mono PCM WAV**. The pipeline converts any input format to this specification automatically via ffmpeg.

**Output format:**
With `--output-json`, the binary writes a JSON file:
```json
{
  "transcription": [
    {
      "timestamps": { "from": "00:00:01,280", "to": "00:00:03,760" },
      "text": "Bonjour tout le monde."
    }
  ]
}
```
Timestamps follow the SRT convention with a **comma** as decimal separator (`HH:MM:SS,mmm`).

**Invocation:**
The pipeline calls the binary as a subprocess and reads the output JSON from disk:
```
whisper-cli -m <model.bin> -f <audio.wav> --language fr --output-json -of <output_base>
```

### faster-whisper (`faster_whisper`)

faster-whisper runs Whisper locally using [CTranslate2](https://github.com/OpenNMT/CTranslate2), a C++ inference engine originally developed for Neural Machine Translation by the OpenNMT team, extended to support encoder-decoder Transformers.

**Model format:**
Original PyTorch weights are converted to CTranslate2's serialization format using `ct2-whisper-converter`. Converted models are hosted on HuggingFace (e.g. `Systran/faster-whisper-base`) and downloaded automatically on first use, cached at `~/.cache/huggingface/hub/`.

**Quantization:**

| `compute_type` | Precision | Use case |
|---|---|---|
| `int8` | 8-bit integer | CPU, best speed/memory tradeoff |
| `float16` | 16-bit float | GPU (CUDA), best throughput |
| `float32` | 32-bit float | Reference, no compression |

`int8` on CPU reduces memory bandwidth by ~4x vs `float32` with negligible accuracy loss on the Whisper architecture.

**Inference:**
The Python API's `WhisperModel.transcribe()` returns a generator of `Segment` objects (`start`, `end`, `text`, and optionally per-word timestamps derived from cross-attention weights). The pipeline consumes this generator to build the final `TranscriptionResult`.

**GPU:**
With `faster_whisper_device: cuda`, CTranslate2 dispatches matrix multiplications to the GPU via cuBLAS. `float16` is recommended on GPU.

### OpenAI API (`openai`)

The API backend delegates inference to OpenAI's servers. Audio chunks are sent over HTTPS and the transcript is returned as JSON. No local compute or model download is required.

**Request format:**
```
POST https://api.openai.com/v1/audio/transcriptions
Authorization: Bearer <api_key>
Content-Type: multipart/form-data

file=<audio_chunk>
model=whisper-1
language=fr            # optional
response_format=verbose_json
```

`verbose_json` returns segments with start/end timestamps, used to produce `srt` and `vtt` outputs.

**Constraints:**
- **25 MB per request**: the chunking pipeline splits audio with ffmpeg before sending, so any file size is handled transparently.
- **Rate limits**: `429 Too Many Requests` responses are handled with exponential backoff.
- The server always runs `whisper-1` (equivalent to `large`). No model size selection is available through the API.
- **Cost:** ~$0.006/min of audio (as of 2024).

### Mega-ASR (`mega_asr`)

An **optional, specialist** backend wrapping [Mega-ASR](https://github.com/xzf-thu/Mega-ASR), a robustness LoRA + router on top of Qwen3-ASR-1.7B, aimed at **heavily degraded audio** (noise, far-field, echo, recording artefacts) where the Whisper backends tend to hallucinate, drop utterances, or return empty output. It is a last-resort backend, not a default.

Two constraints shape how whispr uses it:

- **The LoRA is English/Chinese only.** It was trained on the en/zh Voices-in-the-Wild-2M set, so its robustness gain does not transfer to other languages. whispr mounts the LoRA **only for `en`/`zh`**; for the other ~28 Qwen3-ASR languages it runs the base model (LoRA off) and warns. A language Qwen3-ASR does not support is rejected up front. Mega-ASR's own router keys off *audio quality*, not language, so whispr bypasses it and drives the LoRA from the language instead.
- **It is heavy and GPU-oriented.** A 1.7B model in Transformers on CPU is slow, so CPU is opt-in (`--mega-allow-cpu`). Inference is serialised behind a lock (the wrapper mutates shared LoRA state), so this backend is effectively single-threaded regardless of `--workers`.

**v1 returns text only** (no per-segment timestamps), so `srt`/`vtt` fall back to a single block, so use `txt`/`json`. Timestamps would need Qwen3-ForcedAligner and are left for later.

**Setup** (the `MegaASR` wrapper is not on PyPI; it ships with the repo):
```bash
pip install -r requirements-mega.txt
git clone https://github.com/xzf-thu/Mega-ASR
cd Mega-ASR && python scripts/download.py      # fetches ckpt/Mega-ASR

# then, from whispr, on English degraded audio:
python main.py --file noisy_interview.wav --backend mega_asr --language en \
    --mega-repo /path/to/Mega-ASR --mega-device cuda:0 --format txt,json
```

> Reach for `mega_asr` only on audio the other backends fail on. It also pairs well as a `--fallback-backend` for English jobs.

### Comparison

| | whisper_cpp | faster_whisper | openai | mega_asr |
|---|---|---|---|---|
| **Inference engine** | GGML (C++) | CTranslate2 (C++) | OpenAI server | Transformers (Qwen3-ASR) |
| **Model format** | GGML `.bin` | CTranslate2 (HuggingFace) | Server-side | HuggingFace + LoRA |
| **Quantization** | Static (at conversion) | Dynamic (at load time) | N/A | bf16 (GPU) |
| **Model auto-download** | No | Yes (HuggingFace, first run) | N/A | No (clone + download.py) |
| **Offline** | Yes | Yes | No | Yes |
| **GPU support** | No | Yes (CUDA) | N/A | Yes (recommended) |
| **RAM (base model)** | ~200 MB | ~500 MB | None | ~4 GB (1.7B) |
| **Speed (CPU)** | Fastest | Fast | Network-bound | Slow (opt-in) |
| **Cost** | Free | Free | ~$0.006/min | Free |
| **Privacy** | Full (local) | Full (local) | Audio sent to OpenAI | Full (local) |
| **Setup** | High (compile + model) | Minimal | Minimal (API key) | High (clone + weights) |

| | whisper_cpp | faster_whisper | openai | mega_asr |
|---|---|---|---|---|
| **Best for** | Low RAM machines, offline/embedded, maximum CPU speed | General use, GPU, easy setup | No local compute, one-off jobs | Heavily degraded en/zh audio |
| **Advantages** | Lowest footprint, SIMD-optimised, no Python at runtime | Trivial install, CUDA, int8, word timestamps, active development | Zero local setup, no model management | Robust on noisy/far-field speech, fewer hallucinations |
| **Disadvantages** | Requires compilation, WAV 16 kHz only, no GPU, manual model download | Slower than whisper.cpp on CPU, higher RAM | Internet required, paid, audio leaves your machine, no model choice | GPU + repo clone + multi-GB weights, en/zh LoRA only, no timestamps yet, single-threaded |

**Recommended:** `faster_whisper` for most use cases; `mega_asr` only for degraded English/Chinese audio the others fail on.

---

## Translation

whispr can translate a transcript into another language **fully locally**: no online service, no API key. Translation is powered by [Argos Translate](https://github.com/argosopentech/argos-translate) (OPUS-MT models on the same CTranslate2 engine faster-whisper uses).

It is an **optional extra**, kept out of the core dependencies because it is heavier (it pulls `stanza` → `torch`):

```bash
pip install -r requirements-translate.txt
```

Language models are provisioned on first use: downloaded once from the Argos index, or supplied offline via `--translate-model path/to/model.argosmodel` (pair `--no-translate-download` with it for an air-gapped setup). After that, all translation runs on-device.

### Two modes

```bash
# 1. Transcribe, then translate. The original transcript is kept; a translated
#    copy is written as {prefix}.{lang}.{fmt} (e.g. transcript.fr.txt).
python main.py --file talk.mp4 --language en --translate-to fr --format txt,srt

# 2. Translate an existing text file, no transcription. Source is required
#    because a plain .txt carries no detected language.
python main.py --translate-text transcript.txt --translate-from en --translate-to fr
```

`--translate-to en` reproduces Whisper's translate-to-English behaviour, but any supported pair works in both directions (`--translate-to fr`, `es`, ...), unlike Whisper's English-only `translate` task.

### How it behaves

- **The original is never overwritten.** Translated output is written alongside it with a `.{lang}` suffix.
- **Timestamps are preserved.** For `srt`/`vtt`, each subtitle segment is translated individually so cues stay aligned. The `txt`/`json` body is translated as a whole for better context, so a subtitle line and the plain-text body may be phrased slightly differently. That is expected: alignment for subtitles, context for prose.
- **Graceful when absent.** If `argostranslate` is not installed, a transcription+translation run still writes the transcript and logs a clear warning instead of failing.

> Translation quality is that of the underlying OPUS-MT model for the pair, and is independent of the transcription backend.

---

## Word-level timestamps

By default whispr emits **segment-level** timings. Pass `--word-timestamps` to also get **per-word** timings, stored inside each segment of the JSON output under a `words` list (`{word, start, end, probability}`). It's **opt-in and additive**: without the flag, output is byte-for-byte unchanged; `srt`/`vtt` are untouched either way.

```bash
python main.py --file talk.mp4 --backend faster_whisper --language en \
    --word-timestamps --format json
```

A **provider** selects where the word timing comes from:

| Provider | Source | Deps | Notes |
|---|---|---|---|
| `native` *(default)* | the backend's own word timing | none | faster-whisper (`word_timestamps`), whisper.cpp (`--output-json-full`) |
| `stable_ts` | post-hoc forced alignment (stable-ts) | `requirements-align.txt` | tighter timing than native |
| `whisperx` | post-hoc forced alignment (whisperX, wav2vec2) | `requirements-align.txt` | highest accuracy, heaviest |

```bash
# Higher-accuracy alignment (optional extra: pip install -r requirements-align.txt)
python main.py --file talk.mp4 --language en \
    --word-timestamps --word-timestamps-provider stable_ts --format json
```

- **`native` needs no extra** and is validated end-to-end. It is enough for film / translation subtitle styles.
- **The `stable_ts` and `whisperx` providers are opt-in and heavy** (they pull `torch`). If the package is absent, whispr logs a warning and keeps the timing it already has, so a missing extra never breaks a run. Use them when word timing must land exactly on the syllable (e.g. animated word-by-word captions).
- **`mega_asr` has no timestamps** (v1) and cannot supply word timing.

> This feeds downstream tooling (e.g. styled/animated subtitle burners) that needs to know exactly when each word is spoken.

---

## Web Interface

Same transcription pipeline as the CLI, accessible from a browser.

### Start

```bash
make web

# or manually
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`.

### Features

- Backend detection at startup: green dot if ready, grey with install hint if unavailable
- Auto-selects the best available backend: `faster_whisper` > `whisper_cpp` > `openai`
- Per-backend options shown dynamically:
  - **faster-whisper**: model size dropdown (tiny / base / small / medium / large-v2)
  - **whisper.cpp**: binary and model taken from the server-detected installation
  - **OpenAI**: API key input
- File upload (drag and drop) or URL (YouTube or direct link)
- Language, output formats (TXT / SRT / VTT / JSON), and worker count
- Real-time transcription log via WebSocket
- Download buttons per format on completion

### Exposing it publicly

The server binds to `127.0.0.1` by default. Before putting it on a public
address, terminate TLS on a reverse proxy in front of it and enable the token
gate; the endpoints have no auth otherwise. Configuration is via environment
variables:

| Variable | Default | Purpose |
|---|---|---|
| `WHISPR_AUTH_TOKEN` | *(unset)* | If set, every API/WebSocket call must send it (`X-Whispr-Token` header, `Authorization: Bearer`, or `?token=`). Unset = no auth (local use). |
| `WHISPR_MAX_UPLOAD_MB` | `500` | Upload size ceiling (streamed to disk). |
| `WHISPR_MAX_WORKERS` | `8` | Hard cap on the per-job worker count a request may ask for. |
| `WHISPR_ALLOW_URL_FETCH` | `1` | Set to `0` to disable remote-URL transcription entirely (removes the SSRF surface). |

```bash
export WHISPR_AUTH_TOKEN="$(openssl rand -hex 24)"
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000   # behind your proxy
```

Remote URLs are vetted against non-public address ranges before fetching, but
for defence in depth run the service as an unprivileged user on an
egress-restricted network. whisper.cpp binary/model paths are pinned to what the
server detected and cannot be set from the browser.

---

## The Whisper Model

whispr runs [OpenAI's Whisper](https://openai.com/research/whisper) ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)), an encoder-decoder Transformer for automatic speech recognition, trained on 680,000 hours of weakly-supervised multilingual audio. Audio is resampled to 16 kHz, turned into a log-Mel spectrogram, and decoded in 30-second windows, with segment timestamps predicted inline and the language auto-detected from the first window.

For a full walkthrough of the architecture (preprocessing, encoder, decoder, decoding strategy, timestamp prediction, language detection, training, and limitations), see [RESEARCH_WHISPER.md](RESEARCH_WHISPER.md).

### Model sizes

| Model | Parameters | Layers | Embedding dim | VRAM | Relative speed (CPU) |
|---|---|---|---|---|---|
| `tiny` | 39 M | 4 | 384 | ~1 GB | ~32x |
| `base` | 74 M | 6 | 512 | ~1 GB | ~16x |
| `small` | 244 M | 12 | 768 | ~2 GB | ~6x |
| `medium` | 769 M | 24 | 1024 | ~5 GB | ~2x |
| `large-v2` | 1550 M | 32 | 1280 | ~10 GB | 1x |

Speed is relative to `large-v2` on CPU. All sizes share the same architecture and are interchangeable across backends.

---

## References

[1] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv:2212.04356. https://arxiv.org/abs/2212.04356

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30 (NeurIPS 2017). https://arxiv.org/abs/1706.03762

[3] Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). https://arxiv.org/abs/1508.07909

[4] Klein, G., Kim, Y., Deng, Y., Senellart, J., & Rush, A. M. (2017). *OpenNMT: Open-Source Toolkit for Neural Machine Translation*. Proceedings of ACL 2017, System Demonstrations. https://arxiv.org/abs/1701.02810

[5] Gerganov, G. (2022). *whisper.cpp: Port of OpenAI's Whisper model in C/C++*. GitHub repository. https://github.com/ggerganov/whisper.cpp

[6] Systran. (2023). *faster-whisper: Faster Whisper transcription with CTranslate2*. GitHub repository. https://github.com/SYSTRAN/faster-whisper

[7] Klein, G., Crego, J., & Senellart, J. (2020). *Efficient and High-Quality Neural Machine Translation with OpenNMT*. Proceedings of the 4th Workshop on Neural Generation and Translation. CTranslate2 inference engine underlying faster-whisper. https://github.com/OpenNMT/CTranslate2

[8] OpenAI. (2023). *Whisper API Reference*. OpenAI Platform Documentation. https://platform.openai.com/docs/api-reference/audio

---

## License

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for the full terms.
