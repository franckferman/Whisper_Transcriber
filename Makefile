PYTHON   := python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PY       := $(VENV)/bin/python

# ── Default config/output ─────────────────────────────────────────────────────
CONFIG   ?= config.json
BACKEND  ?= faster_whisper
LANGUAGE ?= fr
WORKERS  ?= 2
OUTPUT   ?= ./output
FILE     ?=
URL      ?=

# Translation (optional; see deps-translate)
TO       ?=
FROM     ?=
IN       ?=

# Mega-ASR (optional; see deps-mega). MEGA_DEVICE empty => CPU (slow, opt-in).
MEGA_REPO   ?=
MEGA_DEVICE ?=

# Web bind. Loopback by default -- put a reverse proxy in front to expose it.
HOST     ?= 127.0.0.1
PORT     ?= 8000

.DEFAULT_GOAL := help

# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: install
install: ## Full installation (venv + deps + config)
	bash install.sh

.PHONY: venv
venv: ## Create virtual environment only
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet

.PHONY: deps
deps: venv ## Install core dependencies
	$(PIP) install --quiet -r requirements.txt

.PHONY: deps-faster-whisper
deps-faster-whisper: deps ## Install faster-whisper backend
	$(PIP) install --quiet faster-whisper

.PHONY: deps-openai
deps-openai: deps ## Install OpenAI backend
	$(PIP) install --quiet openai

.PHONY: deps-translate
deps-translate: deps ## Install local translation (Argos; heavy: pulls torch)
	$(PIP) install --quiet -r requirements-translate.txt

.PHONY: deps-mega
deps-mega: deps ## Install Mega-ASR backend (heavy, GPU-oriented)
	$(PIP) install --quiet -r requirements-mega.txt

.PHONY: deps-align
deps-align: deps ## Install premium word-timestamp providers (stable-ts/whisperx)
	$(PIP) install --quiet -r requirements-align.txt

.PHONY: deps-all
deps-all: deps deps-faster-whisper deps-openai ## Install core + local backends (not the heavy translate/mega extras)

# ── whisper.cpp ───────────────────────────────────────────────────────────────

WHISPER_CPP_DIR   := whisper.cpp
WHISPER_MODEL     ?= base
WHISPER_MODEL_BIN := $(WHISPER_CPP_DIR)/models/ggml-$(WHISPER_MODEL).bin

.PHONY: whisper-cpp-build
whisper-cpp-build: ## Clone and compile whisper.cpp
	@if [ ! -d "$(WHISPER_CPP_DIR)" ]; then \
	    echo "[+] Cloning whisper.cpp..."; \
	    git clone --depth=1 https://github.com/ggerganov/whisper.cpp $(WHISPER_CPP_DIR); \
	else \
	    echo "[+] whisper.cpp already cloned."; \
	fi
	@echo "[+] Compiling..."
	$(MAKE) -C $(WHISPER_CPP_DIR) -j$$(nproc)
	@echo "[+] Build complete."
	@ls $(WHISPER_CPP_DIR)/main $(WHISPER_CPP_DIR)/build/bin/whisper-cli 2>/dev/null | head -1 | xargs -I{} echo "[+] Binary: {}"

.PHONY: whisper-cpp-model
whisper-cpp-model: ## Download whisper.cpp model (WHISPER_MODEL=base|tiny|small|medium|large-v2)
	@if [ -f "$(WHISPER_MODEL_BIN)" ]; then \
	    echo "[+] Model already exists: $(WHISPER_MODEL_BIN)"; \
	else \
	    echo "[+] Downloading model: $(WHISPER_MODEL)..."; \
	    bash $(WHISPER_CPP_DIR)/models/download-ggml-model.sh $(WHISPER_MODEL); \
	fi

.PHONY: whisper-cpp-setup
whisper-cpp-setup: whisper-cpp-build whisper-cpp-model ## Full whisper.cpp setup (build + model)

# ── Run ───────────────────────────────────────────────────────────────────────

.PHONY: run
run: ## Transcribe FILE= or URL= (e.g. make run FILE=video.mp4)
ifndef FILE
ifndef URL
	$(error Specify FILE=path/to/video.mp4 or URL=https://...)
endif
endif
ifdef FILE
	$(PY) main.py --file "$(FILE)" --backend $(BACKEND) --language $(LANGUAGE) \
	    --workers $(WORKERS) --output-dir $(OUTPUT)
endif
ifdef URL
	$(PY) main.py --url "$(URL)" --backend $(BACKEND) --language $(LANGUAGE) \
	    --workers $(WORKERS) --output-dir $(OUTPUT)
endif

.PHONY: run-config
run-config: ## Run using CONFIG= file (default: config.json)
	$(PY) main.py --config $(CONFIG)

.PHONY: dry-run
dry-run: ## Dry-run — show what would be executed without running
ifdef FILE
	$(PY) main.py --file "$(FILE)" --backend $(BACKEND) --dry-run
else ifdef URL
	$(PY) main.py --url "$(URL)" --backend $(BACKEND) --dry-run
else
	$(error Specify FILE= or URL=)
endif

.PHONY: run-whisper-cpp
run-whisper-cpp: ## Run with whisper_cpp backend (auto-detects binary and model)
	$(eval WCPP_BIN := $(shell ls $(WHISPER_CPP_DIR)/build/bin/whisper-cli $(WHISPER_CPP_DIR)/main 2>/dev/null | head -1))
	@if [ -z "$(WCPP_BIN)" ]; then echo "[x] whisper.cpp binary not found. Run: make whisper-cpp-setup"; exit 1; fi
	@if [ ! -f "$(WHISPER_MODEL_BIN)" ]; then echo "[x] Model not found: $(WHISPER_MODEL_BIN). Run: make whisper-cpp-model"; exit 1; fi
ifdef FILE
	$(PY) main.py --file "$(FILE)" --backend whisper_cpp \
	    --whisper-binary "$(WCPP_BIN)" --whisper-model "$(WHISPER_MODEL_BIN)" \
	    --language $(LANGUAGE) --workers $(WORKERS) --output-dir $(OUTPUT)
else ifdef URL
	$(PY) main.py --url "$(URL)" --backend whisper_cpp \
	    --whisper-binary "$(WCPP_BIN)" --whisper-model "$(WHISPER_MODEL_BIN)" \
	    --language $(LANGUAGE) --workers $(WORKERS) --output-dir $(OUTPUT)
else
	$(error Specify FILE= or URL=)
endif

.PHONY: run-faster-whisper
run-faster-whisper: ## Run with faster_whisper backend
	$(MAKE) run BACKEND=faster_whisper

.PHONY: run-openai
run-openai: ## Run with OpenAI API backend
	$(MAKE) run BACKEND=openai

.PHONY: run-mega
run-mega: ## Run mega_asr (MEGA_REPO=/path FILE=... [MEGA_DEVICE=cuda:0])
ifndef FILE
	$(error Specify FILE=path/to/audio.wav)
endif
ifndef MEGA_REPO
	$(error Specify MEGA_REPO=/path/to/Mega-ASR clone -- see 'make deps-mega')
endif
	$(PY) main.py --file "$(FILE)" --backend mega_asr --language $(LANGUAGE) \
	    --mega-repo "$(MEGA_REPO)" \
	    $(if $(MEGA_DEVICE),--mega-device $(MEGA_DEVICE),--mega-allow-cpu) \
	    --workers $(WORKERS) --output-dir $(OUTPUT)

# ── Translation (optional; local, offline) ────────────────────────────────────

.PHONY: run-translate
run-translate: ## Transcribe FILE= then translate to TO= (e.g. make run-translate FILE=talk.mp4 TO=fr)
ifndef FILE
	$(error Specify FILE=path/to/video.mp4)
endif
ifndef TO
	$(error Specify TO=<target ISO 639-1 code>, e.g. TO=fr)
endif
	$(PY) main.py --file "$(FILE)" --backend $(BACKEND) --language $(LANGUAGE) \
	    --translate-to $(TO) $(if $(FROM),--translate-from $(FROM),) \
	    --workers $(WORKERS) --output-dir $(OUTPUT)

.PHONY: translate-text
translate-text: ## Translate an existing text file (IN=notes.txt FROM=en TO=fr)
ifndef IN
	$(error Specify IN=path/to/text.txt)
endif
ifndef FROM
	$(error Specify FROM=<source ISO 639-1 code>, e.g. FROM=en)
endif
ifndef TO
	$(error Specify TO=<target ISO 639-1 code>, e.g. TO=fr)
endif
	$(PY) main.py --translate-text "$(IN)" --translate-from $(FROM) --translate-to $(TO) \
	    --output-dir $(OUTPUT)

# ── Output formats ────────────────────────────────────────────────────────────

.PHONY: run-srt
run-srt: ## Transcribe and output SRT subtitles
ifdef FILE
	$(PY) main.py --file "$(FILE)" --backend $(BACKEND) --language $(LANGUAGE) \
	    --format srt --output-dir $(OUTPUT)
else
	$(error Specify FILE=path/to/video.mp4)
endif

.PHONY: run-all-formats
run-all-formats: ## Transcribe and output txt + json + srt + vtt
ifdef FILE
	$(PY) main.py --file "$(FILE)" --backend $(BACKEND) --language $(LANGUAGE) \
	    --format txt,json,srt,vtt --output-dir $(OUTPUT)
else
	$(error Specify FILE=path/to/video.mp4)
endif

.PHONY: run-words
run-words: ## Transcribe with per-word timestamps (PROVIDER=native|stable_ts|whisperx)
ifndef FILE
	$(error Specify FILE=path/to/video.mp4)
endif
	$(PY) main.py --file "$(FILE)" --backend $(BACKEND) --language $(LANGUAGE) \
	    --word-timestamps $(if $(PROVIDER),--word-timestamps-provider $(PROVIDER),) \
	    --format json --output-dir $(OUTPUT)

# ── Dev ───────────────────────────────────────────────────────────────────────

.PHONY: check
check: ## Validate all imports and CLI
	$(PY) -c "from transcriber.config import TranscriptionConfig; print('config        OK')"
	$(PY) -c "from transcriber.backends.base import TranscriptionBackend; print('backends      OK')"
	$(PY) -c "from transcriber.managers.transcription import TranscriptionManager; print('manager       OK')"
	$(PY) -c "from transcriber.formatters.output import OutputFormatter; print('formatter     OK')"
	$(PY) -c "from transcriber.processors.translate import LocalTranslator; print('translate     OK')"
	$(PY) -c "from transcriber.processors import word_timestamps; print('word_timing   OK')"
	$(PY) -c "from transcriber.backends.mega_asr import MegaAsrBackend; print('mega_asr      OK')"
	$(PY) main.py --help > /dev/null && echo "CLI           OK"

.PHONY: clean
clean: ## Remove output files and temp dirs
	rm -rf output/* __pycache__ transcriber/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

.PHONY: clean-all
clean-all: clean ## Remove venv and output
	rm -rf $(VENV)

# ── Web interface ─────────────────────────────────────────────────────────────

.PHONY: web
web: ## Start web interface (HOST=127.0.0.1 PORT=8000; override to expose)
	$(PIP) install --quiet fastapi "uvicorn[standard]" python-multipart
	$(PY) -m uvicorn web.app:app --host $(HOST) --port $(PORT)

.PHONY: web-install
web-install: ## Install web dependencies only
	$(PIP) install --quiet fastapi "uvicorn[standard]" python-multipart

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@echo ""
	@echo "  Whisper Transcriber"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Variables (override with make VAR=value):"
	@echo "    BACKEND    $(BACKEND)"
	@echo "    LANGUAGE   $(LANGUAGE)"
	@echo "    WORKERS    $(WORKERS)"
	@echo "    OUTPUT     $(OUTPUT)"
	@echo "    TO / FROM  translation target / source language (ISO 639-1)"
	@echo "    IN         text file to translate (translate-text)"
	@echo "    MEGA_REPO  path to a Mega-ASR clone   MEGA_DEVICE  cuda:0 | mps (empty => CPU)"
	@echo ""
	@echo "  Examples:"
	@echo "    make run FILE=video.mp4"
	@echo "    make run URL=https://youtube.com/... BACKEND=whisper_cpp LANGUAGE=en"
	@echo "    make run-all-formats FILE=video.mp4"
	@echo "    make run-translate FILE=talk.mp4 LANGUAGE=en TO=fr"
	@echo "    make translate-text IN=notes.txt FROM=en TO=fr"
	@echo "    make run-mega FILE=noisy.wav LANGUAGE=en MEGA_REPO=/path/to/Mega-ASR"
	@echo "    make dry-run URL=https://..."
	@echo "    make check"
	@echo ""
