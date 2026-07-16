<div align="center">

# whispr

**A modular Python CLI and web UI for transcribing audio and video with multi-backend support.**

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Backends](https://img.shields.io/badge/backends-whisper.cpp_%7C_faster--whisper_%7C_openai-555555?style=flat-square)](https://github.com/franckferman/whispr)

</div>

---

## Table of Contents

- [Overview](#overview)
- [The Whisper Model](#the-whisper-model)
- [Backends](#backends)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Web Interface](#web-interface)
- [Makefile Reference](#makefile-reference)
- [References](#references)
- [License](#license)

---

## Overview

whispr is a transcription pipeline built around OpenAI's Whisper model. It supports three interchangeable backends, parallel chunk processing, automatic fallback, and multiple output formats.

Two interfaces ship with the same codebase:

- A **CLI** (`main.py`) for scripting, automation, and direct use.
- A **web UI** (`web/`) for browser-based use — upload a file or paste a URL, pick a backend, download the result.

**Key capabilities:**

| Area | Details |
|---|---|
| **Backends** | whisper.cpp (GGML subprocess), faster-whisper (CTranslate2), OpenAI Whisper API |
| **Input** | Local files, YouTube URLs (yt-dlp), direct HTTP URLs |
| **Input formats** | MP3, MP4, WAV, M4A, MKV, WEBM, OGG, FLAC, AVI, MOV, and any format supported by ffmpeg |
| **Output formats** | Plain text, JSON (with segments and timestamps), SRT subtitles, WebVTT subtitles |
| **Chunking** | ffmpeg-based splitting for arbitrarily long files, transparent to the user |
| **Parallel processing** | ThreadPoolExecutor, configurable worker count |
| **Retry / fallback** | Exponential backoff per chunk, automatic switch to secondary backend on failure |
| **Configuration** | JSON file + CLI overrides + `${ENV_VAR}` interpolation |
| **Web interface** | FastAPI + vanilla JS, real-time WebSocket log, per-backend options |

---

## The Whisper Model

[Whisper](https://openai.com/research/whisper) is an automatic speech recognition (ASR) model published by OpenAI in September 2022 ([Radford et al., arXiv:2212.04356](https://arxiv.org/abs/2212.04356)). It is a sequence-to-sequence model based on a standard **encoder-decoder Transformer** architecture, trained end-to-end with weak supervision on a large-scale multilingual dataset.

### Audio preprocessing

Raw audio undergoes the following transformations before reaching the model:

1. Resampled to **16 kHz mono**.
2. A **log-Mel spectrogram** is computed with a 25 ms window, 10 ms stride, and **80 Mel frequency bins**.
3. The spectrogram is split into **30-second windows** of 3000 frames (padded with silence if shorter).

The resulting 80 x 3000 tensor is the input to the encoder.

### Encoder

The encoder consists of a convolutional front-end followed by Transformer blocks:

- Two 1D convolution layers (kernel size 3, GELU activation) downsample the time axis by a factor of 2, reducing the sequence from 3000 to 1500 frames.
- A stack of Transformer encoder blocks with multi-head self-attention and learned sinusoidal positional embeddings produces a sequence of contextual hidden representations.

Hyperparameters scale with model size (e.g. `base`: 6 layers, 512 dims, 8 heads — `large-v2`: 32 layers, 1280 dims, 20 heads).

### Decoder

The decoder generates text tokens autoregressively via cross-attention over the encoder output:

- A multilingual **Byte-Pair Encoding (BPE)** tokenizer with a vocabulary of ~50,000 tokens.
- Each generation is conditioned on a structured prompt of special tokens:
  - `<|startoftranscript|>` — sequence boundary
  - `<|fr|>`, `<|en|>`, ... — target language (omitted for auto-detect)
  - `<|transcribe|>` or `<|translate|>` — task selector
  - `<|notimestamps|>` or `<|0.00|>` — disables or enables timestamp prediction

### Timestamp prediction

When timestamp mode is active, the decoder interleaves **timestamp tokens** (`<|0.00|>` to `<|30.00|>`, 0.02 s resolution) with text tokens, enabling segment-level timing without any external alignment step.

### Language detection

When no language is specified, Whisper runs the encoder on the first 30 seconds, then takes a softmax over language tokens at the first decoder step to identify the language. This adds ~1-2 seconds of overhead and works reliably for the 99 languages present in the training data.

### Training

Whisper was trained on **680,000 hours** of multilingual audio paired with transcripts collected from the web. Training uses standard cross-entropy over token sequences. The dataset was not manually curated — transcripts were obtained automatically (weak supervision), which accounts for the model's breadth as well as its sensitivity to low-resource languages and strong accents.

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

## Backends

The three backends execute Whisper inference through different runtimes, model formats, and compute environments. Transcription quality is identical for the same model size.

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
- **25 MB per request** — the chunking pipeline splits audio with ffmpeg before sending, so any file size is handled transparently.
- **Rate limits** — `429 Too Many Requests` responses are handled with exponential backoff.
- The server always runs `whisper-1` (equivalent to `large`). No model size selection is available through the API.
- **Cost:** ~$0.006/min of audio (as of 2024).

### faster-whisper (`faster_whisper`)

faster-whisper runs Whisper locally using [CTranslate2](https://github.com/OpenNMT/CTranslate2), a C++ inference engine originally developed for Neural Machine Translation by the OpenNMT team, extended to support encoder-decoder Transformers.

**Model format:**
Original PyTorch weights are converted to CTranslate2's serialization format using `ct2-whisper-converter`. Converted models are hosted on HuggingFace (e.g. `Systran/faster-whisper-base`) and downloaded automatically on first use, cached at `~/.cache/huggingface/hub/`.

**Quantization:**

| `compute_type` | Precision | Use case |
|---|---|---|
| `int8` | 8-bit integer | CPU — best speed/memory tradeoff |
| `float16` | 16-bit float | GPU (CUDA) — best throughput |
| `float32` | 32-bit float | Reference, no compression |

`int8` on CPU reduces memory bandwidth by ~4x vs `float32` with negligible accuracy loss on the Whisper architecture.

**Inference:**
The Python API's `WhisperModel.transcribe()` returns a generator of `Segment` objects (`start`, `end`, `text`, and optionally per-word timestamps derived from cross-attention weights). The pipeline consumes this generator to build the final `TranscriptionResult`.

**GPU:**
With `faster_whisper_device: cuda`, CTranslate2 dispatches matrix multiplications to the GPU via cuBLAS. `float16` is recommended on GPU.

### whisper.cpp (`whisper_cpp`)

whisper.cpp is a standalone C++ reimplementation of Whisper by [Georgi Gerganov](https://github.com/ggerganov/whisper.cpp), built on [GGML](https://github.com/ggerganov/ggml) — a minimal C tensor library with no dependencies beyond the C standard library.

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

### Comparison

| | whisper_cpp | faster_whisper | openai |
|---|---|---|---|
| **Inference engine** | GGML (C++) | CTranslate2 (C++) | OpenAI server |
| **Model format** | GGML `.bin` | CTranslate2 (HuggingFace) | Server-side |
| **Quantization** | Static (at conversion) | Dynamic (at load time) | N/A |
| **Model auto-download** | No | Yes (HuggingFace, first run) | N/A |
| **Offline** | Yes | Yes | No |
| **GPU support** | No | Yes (CUDA) | N/A |
| **RAM (base model)** | ~200 MB | ~500 MB | None |
| **Speed (CPU)** | Fastest | Fast | Network-bound |
| **Cost** | Free | Free | ~$0.006/min |
| **Privacy** | Full (local) | Full (local) | Audio sent to OpenAI |
| **Setup** | High (compile + model) | Minimal | Minimal (API key) |

| | whisper_cpp | faster_whisper | openai |
|---|---|---|---|
| **Best for** | Low RAM machines, offline/embedded, maximum CPU speed | General use, GPU, easy setup | No local compute, one-off jobs |
| **Advantages** | Lowest footprint, SIMD-optimised, no Python at runtime | Trivial install, CUDA, int8, word timestamps, active development | Zero local setup, no model management |
| **Disadvantages** | Requires compilation, WAV 16 kHz only, no GPU, manual model download | Slower than whisper.cpp on CPU, higher RAM | Internet required, paid, audio leaves your machine, no model choice |

**Recommended:** `faster_whisper` for most use cases.

---

## Project Structure

```
whispr/
├── main.py                        # CLI entry point (argparse)
├── config.example.json            # Annotated configuration reference
├── requirements.txt               # Core dependencies (optional deps commented)
├── install.sh                     # Interactive setup script
├── Makefile                       # Common tasks
│
├── transcriber/
│   ├── config.py                  # Config dataclass, JSON loading, env interpolation
│   ├── logger.py                  # Structured logging (file + stderr)
│   ├── backends/
│   │   ├── base.py                # TranscriptionBackend ABC + TranscriptionResult
│   │   ├── whisper_cpp.py         # subprocess to whisper.cpp binary
│   │   ├── faster_whisper.py      # faster-whisper (optional import)
│   │   └── openai_api.py          # OpenAI Whisper API with rate-limit handling
│   ├── processors/
│   │   ├── video.py               # Source resolution: local / YouTube / HTTP
│   │   └── audio.py               # WAV conversion, duration probing
│   ├── managers/
│   │   └── transcription.py       # Orchestrator: chunks, parallel, retry, merge
│   └── formatters/
│       └── output.py              # txt, json, srt, vtt writers
│
└── web/
    ├── app.py                     # FastAPI application
    └── static/
        ├── index.html
        ├── style.css
        └── app.js
```

---

## Installation

**System prerequisites:** Python 3.9+, ffmpeg

ffmpeg is required at the system level for audio extraction, format conversion, and chunk splitting. Install it with your package manager if not already present (`apt-get install ffmpeg`, `brew install ffmpeg`, `dnf install ffmpeg`). `install.sh` checks for it and installs it automatically if absent.

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

What the installer does:
1. Verifies Python 3.9+ and ffmpeg (installs ffmpeg if absent)
2. Creates a virtual environment in `.venv/`
3. Installs core dependencies (`ffmpeg-python`, `tqdm`, `requests`, `yt-dlp`, `pydub`)
4. Installs web UI dependencies if selected
5. Installs the chosen backend(s)
6. For whisper.cpp: clones the repository, compiles with `make -j$(nproc)`, prompts for model download, and writes the detected binary and model paths into `config.json`
7. Copies `config.example.json` to `config.json` if absent

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
| `backend` | `faster_whisper` | Primary backend: `faster_whisper`, `whisper_cpp`, `openai` |
| `fallback_backend` | `null` | Secondary backend used if primary exhausts all retries |
| `faster_whisper_model` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v2` |
| `faster_whisper_device` | `cpu` | Compute device: `cpu` or `cuda` |
| `faster_whisper_compute_type` | `int8` | Quantization: `int8`, `float16`, `float32` |
| `whisper_cpp_binary` | `whisper` | Path to the compiled whisper.cpp binary |
| `whisper_cpp_model` | `models/ggml-base.bin` | Path to the GGML `.bin` model file |
| `whisper_cpp_extra_args` | `[]` | Additional CLI arguments forwarded verbatim to the binary |
| `openai_api_key` | `${OPENAI_API_KEY}` | OpenAI API key |
| `openai_model` | `whisper-1` | OpenAI model identifier |
| `language` | `null` | ISO 639-1 code (`fr`, `en`, ...) or `null` for auto-detect |
| `chunk_duration_seconds` | `600` | Duration of each audio chunk in seconds |
| `workers` | `2` | Number of parallel transcription threads |
| `temp_dir` | `null` | Temporary directory (system default if null) |
| `output_formats` | `["txt"]` | Output formats: `txt`, `json`, `srt`, `vtt` |
| `output_dir` | `.` | Directory where output files are written |
| `output_prefix` | `transcript` | Filename prefix for output files |
| `max_retries` | `3` | Retry attempts per chunk before declaring failure |
| `retry_base_delay` | `1.0` | Base delay in seconds for exponential backoff |
| `retry_max_delay` | `30.0` | Maximum backoff delay cap in seconds |
| `dry_run` | `false` | Print the execution plan without running anything |
| `debug` | `false` | Enable DEBUG-level logging |
| `log_file` | `null` | Write logs to a file (`auto` generates a timestamped filename) |
| `ytdlp_format` | `bestaudio/best` | yt-dlp format selector |
| `ytdlp_output_template` | `%(title)s.%(ext)s` | yt-dlp output filename template |

---

## CLI Usage

```
python main.py [--file PATH | --url URL] [options]
```

If `--backend` is not specified, the value from `config.json` is used (default: `faster_whisper`). The CLI does not perform automatic backend detection — set your preferred backend in `config.json` once and omit the flag on subsequent runs.

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

# Load config from file
python main.py --config config.json
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--file`, `-f` | | Local audio/video file |
| `--url`, `-u` | | YouTube URL or direct HTTP link |
| `--config`, `-c` | | JSON config file (CLI flags override) |
| `--backend`, `-b` | `faster_whisper` | `whisper_cpp` \| `faster_whisper` \| `openai` |
| `--fallback-backend` | | Secondary backend if primary fails all retries |
| `--whisper-binary` | `whisper` | Path to the whisper.cpp binary |
| `--whisper-model` | | Path to the GGML `.bin` model file |
| `--fw-model` | `base` | faster-whisper model size |
| `--fw-device` | `cpu` | faster-whisper device: `cpu` \| `cuda` |
| `--openai-key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--language`, `-l` | auto-detect | ISO 639-1 code: `fr`, `en`, `es`, ... |
| `--chunk-duration` | `600` | Chunk size in seconds |
| `--workers`, `-w` | `2` | Parallel transcription threads |
| `--format`, `-F` | `txt` | Output formats, comma-separated: `txt,json,srt,vtt` |
| `--output-dir`, `-o` | `.` | Output directory |
| `--output-prefix` | `transcript` | Output filename prefix |
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
| `srt` | SubRip subtitles — VLC, ffmpeg, Premiere |
| `vtt` | WebVTT — HTML5 `<video>` element |

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

- Backend detection at startup — green dot if ready, grey with install hint if unavailable
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
gate — the endpoints have no auth otherwise. Configuration is via environment
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

## Makefile Reference

```bash
make install              # Full setup via install.sh
make deps                 # Core Python deps only
make deps-faster-whisper  # + faster-whisper
make deps-openai          # + openai
make deps-all             # All Python deps

make whisper-cpp-build    # Clone and compile whisper.cpp
make whisper-cpp-model    # Download model (WHISPER_MODEL=base|tiny|small|medium|large-v2)
make whisper-cpp-setup    # Build + model in one step

make run FILE=video.mp4                         # Transcribe (faster_whisper, fr by default)
make run URL=https://...                        # Transcribe a URL
make run FILE=video.mp4 BACKEND=whisper_cpp     # Specific backend
make run FILE=video.mp4 LANGUAGE=en WORKERS=4  # Override defaults
make run-whisper-cpp FILE=video.mp4
make run-faster-whisper FILE=video.mp4
make run-openai FILE=video.mp4

make run-srt FILE=video.mp4                     # SRT output only
make run-all-formats FILE=video.mp4             # txt + json + srt + vtt

make dry-run FILE=video.mp4                     # Preview without executing
make run-config                                 # Use config.json

make web                                        # Start web UI at http://localhost:8000
make web-install                                # Install web deps only

make check                                      # Validate imports + CLI
make clean                                      # Remove output files and __pycache__
make clean-all                                  # Remove venv too
```

---

## References

[1] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv:2212.04356. https://arxiv.org/abs/2212.04356

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30 (NeurIPS 2017). https://arxiv.org/abs/1706.03762

[3] Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). https://arxiv.org/abs/1508.07909

[4] Klein, G., Kim, Y., Deng, Y., Senellart, J., & Rush, A. M. (2017). *OpenNMT: Open-Source Toolkit for Neural Machine Translation*. Proceedings of ACL 2017, System Demonstrations. https://arxiv.org/abs/1701.02810

[5] Gerganov, G. (2022). *whisper.cpp: Port of OpenAI's Whisper model in C/C++*. GitHub repository. https://github.com/ggerganov/whisper.cpp

[6] Systran. (2023). *faster-whisper: Faster Whisper transcription with CTranslate2*. GitHub repository. https://github.com/SYSTRAN/faster-whisper

[7] Klein, G., Crego, J., & Senellart, J. (2020). *Efficient and High-Quality Neural Machine Translation with OpenNMT*. Proceedings of the 4th Workshop on Neural Generation and Translation. — CTranslate2 inference engine underlying faster-whisper. https://github.com/OpenNMT/CTranslate2

[8] OpenAI. (2023). *Whisper API Reference*. OpenAI Platform Documentation. https://platform.openai.com/docs/api-reference/audio

---

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

Any use, modification, or distribution — including over a network — requires the full source code to remain open under the same license.
