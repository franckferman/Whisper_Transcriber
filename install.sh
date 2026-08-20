#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[+]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[x]${NC} $*" >&2; exit 1; }

# Tracks what was installed for the final summary
INSTALLED_BINARY=""
INSTALLED_MODEL=""

_download_model() {
    local model_name="$1"
    if [[ -f "whisper.cpp/models/ggml-${model_name}.bin" ]]; then
        info "Model already exists: whisper.cpp/models/ggml-${model_name}.bin"
    else
        info "Downloading model: $model_name..."
        bash whisper.cpp/models/download-ggml-model.sh "$model_name"
        info "Model saved to: whisper.cpp/models/ggml-${model_name}.bin"
    fi
}

_install_whisper_cpp() {
    info "Setting up whisper.cpp..."

    # Build deps
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y git build-essential cmake 2>/dev/null || true
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y git gcc gcc-c++ make cmake 2>/dev/null || true
    elif command -v brew &>/dev/null; then
        brew install cmake 2>/dev/null || true
    fi

    if [[ ! -d "whisper.cpp" ]]; then
        info "Cloning whisper.cpp..."
        git clone --depth=1 https://github.com/ggerganov/whisper.cpp
    else
        info "whisper.cpp already cloned."
    fi

    info "Compiling whisper.cpp (make -j$(nproc))..."
    make -C whisper.cpp -j"$(nproc)" 2>&1 | tail -5

    # Locate binary (path varies across whisper.cpp versions)
    local binary=""
    for candidate in whisper.cpp/build/bin/whisper-cli whisper.cpp/main whisper.cpp/whisper-cli; do
        if [[ -x "$candidate" ]]; then
            binary="$candidate"
            break
        fi
    done

    if [[ -z "$binary" ]]; then
        warn "Could not locate whisper.cpp binary after build. Check whisper.cpp/ manually."
    else
        info "Binary found: $binary"
        INSTALLED_BINARY="$binary"
    fi

    # Download one or more models
    echo ""
    echo "Available models (larger = better quality, slower):"
    echo "  [1] tiny     (~75MB)   — fastest"
    echo "  [2] base     (~150MB)  — good balance (recommended)"
    echo "  [3] small    (~500MB)"
    echo "  [4] medium   (~1.5GB)"
    echo "  [5] large-v2 (~3GB)   — best quality"
    echo ""
    echo "You can download multiple models (run this loop several times)."
    echo ""

    local last_model=""
    while true; do
        read -rp "Download model? [1-5] (Enter to finish): " MODEL_CHOICE
        local model_name=""
        case "$MODEL_CHOICE" in
            1) model_name="tiny"     ;;
            2) model_name="base"     ;;
            3) model_name="small"    ;;
            4) model_name="medium"   ;;
            5) model_name="large-v2" ;;
            "") break ;;
            *) warn "Invalid choice, skipping."; continue ;;
        esac
        _download_model "$model_name"
        last_model="$model_name"
    done

    if [[ -z "$last_model" ]]; then
        warn "No model downloaded. Run: bash whisper.cpp/models/download-ggml-model.sh base"
    else
        INSTALLED_MODEL="$last_model"
    fi

    # Update config.json with detected binary and last downloaded model
    if [[ -f "config.json" && -n "$binary" && -n "$last_model" ]]; then
        python3 - <<PYEOF
import json
with open('config.json') as f:
    cfg = json.load(f)
cfg['whisper_cpp_binary'] = './$binary'
cfg['whisper_cpp_model']  = './whisper.cpp/models/ggml-${last_model}.bin'
with open('config.json', 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write('\n')
PYEOF
        info "config.json updated: whisper_cpp_binary and whisper_cpp_model set."
    fi

    # Print usage hint
    if [[ -n "$binary" && -n "$last_model" ]]; then
        echo ""
        info "whisper.cpp ready. Usage:"
        echo "  python main.py --file video.mp4 --backend whisper_cpp \\"
        echo "      --whisper-binary ./$binary \\"
        echo "      --whisper-model ./whisper.cpp/models/ggml-${last_model}.bin"
    fi
}

# ── Python ──────────────────────────────────────────────────────────────────
info "Checking Python..."
$PYTHON -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'" \
    || error "Python 3.10+ required. Set PYTHON= to override."
PY_VER=$($PYTHON --version)
info "Found $PY_VER"

# ── ffmpeg ───────────────────────────────────────────────────────────────────
info "Checking ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    info "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    warn "ffmpeg not found. Installing..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y ffmpeg
    elif command -v brew &>/dev/null; then
        brew install ffmpeg
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y ffmpeg
    else
        error "Cannot install ffmpeg automatically. Install it manually: https://ffmpeg.org"
    fi
fi

# ── venv ─────────────────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
else
    info "Virtual environment already exists."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── Core deps ────────────────────────────────────────────────────────────────
info "Installing core dependencies..."
pip install --quiet \
    "ffmpeg-python>=0.2.0" \
    "tqdm>=4.67.0" \
    "requests>=2.28.0" \
    "yt-dlp>=2024.1.0" \
    "pydub>=0.25.1"

# ── Interface ─────────────────────────────────────────────────────────────────
echo ""
echo "Interface:"
echo "  [1] CLI only"
echo "  [2] Web UI  (adds fastapi, uvicorn, python-multipart)"
echo "  [3] Both"
echo ""
read -rp "Which interface? [1/2/3]: " IFACE_CHOICE

case "$IFACE_CHOICE" in
    2|3)
        info "Installing web UI dependencies..."
        pip install --quiet "fastapi>=0.100.0" "uvicorn[standard]>=0.23.0" "python-multipart>=0.0.6"
        ;;
esac

# ── Optional backends ─────────────────────────────────────────────────────────
echo ""
echo "Optional backends:"
echo "  [1] faster-whisper  (local, no API key, recommended)"
echo "  [2] OpenAI API      (cloud, requires OPENAI_API_KEY)"
echo "  [3] whisper.cpp     (local C++ binary, fastest on CPU — requires build)"
echo "  [4] All"
echo "  [5] Skip"
echo ""
read -rp "Install optional backend(s)? [1/2/3/4/5]: " CHOICE

case "$CHOICE" in
    1|4)
        info "Installing faster-whisper..."
        pip install --quiet "faster-whisper>=0.10.0"
        ;;
esac

case "$CHOICE" in
    2|4)
        info "Installing openai..."
        pip install --quiet "openai>=1.0.0"
        ;;
esac

case "$CHOICE" in
    3|4)
        _install_whisper_cpp
        ;;
esac

# ── Config ───────────────────────────────────────────────────────────────────
if [[ ! -f "config.json" ]]; then
    info "Creating config.json from example..."
    cp config.example.json config.json
    warn "Edit config.json to set your paths and API keys."
fi

mkdir -p output

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Installation complete."
echo ""
echo "  Activate:  source $VENV_DIR/bin/activate"

if [[ "$CHOICE" == "3" && -n "$INSTALLED_BINARY" && -n "$INSTALLED_MODEL" ]]; then
    echo "  Run:       python main.py --help"
    echo "  Example:   python main.py --file video.mp4 --backend whisper_cpp \\"
    echo "                 --whisper-binary ./$INSTALLED_BINARY \\"
    echo "                 --whisper-model ./whisper.cpp/models/ggml-${INSTALLED_MODEL}.bin"
elif [[ "$CHOICE" == "1" || "$CHOICE" == "4" ]]; then
    echo "  Run:       python main.py --help"
    echo "  Example:   python main.py --file video.mp4 --backend faster_whisper --language fr"
else
    echo "  Run:       python main.py --help"
fi

if [[ "${IFACE_CHOICE:-1}" == "2" || "${IFACE_CHOICE:-1}" == "3" ]]; then
    echo "  Web UI:    python -m uvicorn web.app:app --host 127.0.0.1 --port 8000"
    echo "  Open:      http://localhost:8000"
    echo "  Public:    put a TLS reverse proxy in front and set WHISPR_AUTH_TOKEN"
fi
echo ""
