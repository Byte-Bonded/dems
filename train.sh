#!/usr/bin/env bash
# =============================================================================
#  DEMS — Automated Hierarchical Multi-Agent PPO Training (GPU-Accelerated)
# =============================================================================
#  Usage:
#    ./train.sh                   # default 50K steps (auto-detect GPU)
#    ./train.sh quick             # smoke-test (500 steps)
#    ./train.sh full              # full training (500K steps)
#    ./train.sh custom 100000 96  # custom steps + episode length
#
#  Environment variables:
#    DEVICE=cuda|cpu|mps|auto     # force device (default: auto)
#    SEED=42                      # random seed
#    BATCH_SIZE=256               # override batch size
#    NO_GPU=1                     # force CPU training
# =============================================================================
set -euo pipefail

# ── Project paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Auto-detect venv: check ./venv, ./.venv, ../venv, ../.venv, or VENV_DIR env var
if [[ -n "${VENV_DIR:-}" ]]; then
    :  # user-supplied
elif [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
    VENV_DIR="$PROJECT_ROOT/venv"
elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    VENV_DIR="$PROJECT_ROOT/.venv"
elif [[ -f "$(dirname "$PROJECT_ROOT")/venv/bin/activate" ]]; then
    VENV_DIR="$(dirname "$PROJECT_ROOT")/venv"
elif [[ -f "$(dirname "$PROJECT_ROOT")/.venv/bin/activate" ]]; then
    VENV_DIR="$(dirname "$PROJECT_ROOT")/.venv"
else
    VENV_DIR=""  # no venv found, use system Python
fi

LOG_DIR="$PROJECT_ROOT/logs/rl"
TRAIN_LOG="$PROJECT_ROOT/logs/train_$(date +%Y%m%d_%H%M%S).log"

# ── Training presets ─────────────────────────────────────────────────────────
PRESET="${1:-default}"
case "$PRESET" in
    quick|smoke)
        STEPS=500
        EPISODE_LEN=12
        EVAL_FREQ=250
        DESC="Smoke-test"
        ;;
    default|medium)
        STEPS=50000
        EPISODE_LEN=48
        EVAL_FREQ=5000
        DESC="Default (50K steps)"
        ;;
    full|long)
        STEPS=500000
        EPISODE_LEN=48
        EVAL_FREQ=25000
        DESC="Full training (500K steps)"
        ;;
    custom)
        STEPS="${2:?Usage: ./train.sh custom <steps> [episode_len]}"
        EPISODE_LEN="${3:-48}"
        EVAL_FREQ=$(( STEPS / 10 ))
        DESC="Custom ($STEPS steps)"
        ;;
    *)
        echo "Unknown preset: $PRESET"
        echo "Usage: ./train.sh [quick|default|full|custom <steps> [ep_len]]"
        exit 1
        ;;
esac

SEED="${SEED:-42}"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
fail() { echo -e "${RED}[FAILED]${NC} $*"; exit 1; }

# ── Banner ───────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   DEMS — Hierarchical Multi-Agent PPO Training           ║"
echo "║   117-Bus Tri-Area SuperGrid  |  13 PPO Agents           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Activate virtual environment ─────────────────────────────────────────
if [[ -n "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    ok "Virtual environment activated: $VENV_DIR"
else
    warn "No virtual environment found — using system Python"
    warn "Create one with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
fi

# ── 2. Verify dependencies ──────────────────────────────────────────────────
cd "$PROJECT_ROOT"
python3 -c "import pandapower, stable_baselines3, gymnasium, torch, numpy" 2>/dev/null \
    || fail "Missing dependencies. Run: pip install -r requirements.txt"
ok "Dependencies verified"

# ── 3. GPU Detection ────────────────────────────────────────────────────────
if [[ "${NO_GPU:-0}" == "1" ]]; then
    DEVICE="cpu"
    GPU_NAME="(disabled by NO_GPU=1)"
    GPU_MEM_GB="0"
else
    DEVICE="${DEVICE:-auto}"
fi

# Run Python GPU detection to get full details
GPU_INFO=$(python3 -c "
import torch, json
info = {'device': 'cpu', 'gpu_name': 'N/A', 'gpu_mem_gb': 0, 'cuda_version': 'N/A', 'torch_version': torch.__version__}
requested = '${DEVICE}'
if requested == 'auto':
    if torch.cuda.is_available():
        info['device'] = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device'] = 'mps'
else:
    info['device'] = requested
if info['device'] == 'cuda' and torch.cuda.is_available():
    info['gpu_name'] = torch.cuda.get_device_name(0)
    info['gpu_mem_gb'] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
    info['cuda_version'] = torch.version.cuda or 'N/A'
    info['cudnn_version'] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else 'N/A'
    info['compute_capability'] = '.'.join(str(x) for x in torch.cuda.get_device_capability(0))
elif info['device'] == 'mps':
    info['gpu_name'] = 'Apple Metal (MPS)'
print(json.dumps(info))
" 2>/dev/null) || GPU_INFO='{"device":"cpu","gpu_name":"N/A","gpu_mem_gb":0,"torch_version":"?","cuda_version":"N/A"}'

DEVICE=$(echo "$GPU_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['device'])")
GPU_NAME=$(echo "$GPU_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['gpu_name'])")
GPU_MEM_GB=$(echo "$GPU_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['gpu_mem_gb'])")
TORCH_VER=$(echo "$GPU_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['torch_version'])")
CUDA_VER=$(echo "$GPU_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cuda_version','N/A'))")

if [[ "$DEVICE" == "cuda" ]]; then
    echo -e "${GREEN}${BOLD}  ┌──────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}${BOLD}  │  GPU DETECTED: $GPU_NAME${NC}"
    echo -e "${GREEN}${BOLD}  │  VRAM: ${GPU_MEM_GB} GB | CUDA: ${CUDA_VER} | PyTorch: ${TORCH_VER}${NC}"
    echo -e "${GREEN}${BOLD}  └──────────────────────────────────────────────────┘${NC}"
    ok "CUDA GPU acceleration enabled"

    # ── CUDA performance tuning ──────────────────────────────────────
    export CUDA_LAUNCH_BLOCKING=0                                    # async GPU ops
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"        # reduce VRAM fragmentation
    export CUBLAS_WORKSPACE_CONFIG=":4096:8"                         # deterministic cuBLAS
    export TF_CPP_MIN_LOG_LEVEL=3                                    # suppress TF warnings

    # Enable TF32 for Ampere+ GPUs (RTX 3000/4000 series) — ~2x matmul speedup
    export NVIDIA_TF32_OVERRIDE=1

    # VRAM hard cap — leave ~1 GB headroom for OS/display
    VRAM_LIMIT="${VRAM_LIMIT:-7.0}"
    log "VRAM cap: ${VRAM_LIMIT} GB (of ${GPU_MEM_GB} GB total)"

elif [[ "$DEVICE" == "mps" ]]; then
    echo -e "${GREEN}${BOLD}  ┌──────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}${BOLD}  │  Apple Metal (MPS) backend detected              │${NC}"
    echo -e "${GREEN}${BOLD}  │  PyTorch: ${TORCH_VER}                                     │${NC}"
    echo -e "${GREEN}${BOLD}  └──────────────────────────────────────────────────┘${NC}"
    ok "MPS GPU acceleration enabled"
else
    warn "No GPU detected — training on CPU (will be slower)"
fi
echo ""

# ── 4. Set batch sizes based on device ───────────────────────────────────────
# RTX 4060 (8GB VRAM) can handle larger batches → better GPU utilisation
if [[ -n "${BATCH_SIZE:-}" ]]; then
    BATCH_ARGS="--batch-size $BATCH_SIZE"
    log "Batch size override: $BATCH_SIZE"
else
    BATCH_ARGS=""
fi

# ── Print config ─────────────────────────────────────────────────────────────
log "Preset:         ${BOLD}$DESC${NC}"
log "Steps:          $STEPS"
log "Episode length: $EPISODE_LEN"
log "Eval frequency: $EVAL_FREQ"
log "Seed:           $SEED"
log "Device:         ${BOLD}$DEVICE${NC}"
if [[ "$DEVICE" == "cuda" ]]; then
    log "GPU:            ${GREEN}$GPU_NAME (${GPU_MEM_GB} GB)${NC}"
fi
log "Log file:       $TRAIN_LOG"
echo ""

# ── 5. Quick import check ───────────────────────────────────────────────────
python3 -c "from src.agent.training.ctde_trainer import HierarchicalTrainer" 2>/dev/null \
    || fail "Cannot import HierarchicalTrainer — check src/ for errors"
ok "Project imports OK"

# ── 6. Run tests (skip on smoke-test) ───────────────────────────────────────
if [[ "$PRESET" != "quick" && "$PRESET" != "smoke" ]]; then
    log "Running test sanity check..."
    if python3 -m pytest tests/ -x -q --tb=line 2>&1 | tail -3; then
        ok "Tests passed"
    else
        warn "Some tests failed — training will proceed anyway"
    fi
fi

# ── 7. Backup previous models ───────────────────────────────────────────────
if [[ -d "$LOG_DIR/final" ]]; then
    BACKUP="$LOG_DIR/backup_$(date +%Y%m%d_%H%M%S)"
    log "Backing up previous models to $BACKUP"
    mkdir -p "$BACKUP"
    mv "$LOG_DIR/best" "$BACKUP/best" 2>/dev/null || true
    mv "$LOG_DIR/final" "$BACKUP/final" 2>/dev/null || true
    ok "Previous models backed up"
fi

mkdir -p "$LOG_DIR" "$(dirname "$TRAIN_LOG")"

# ── 8. GPU warm-up (pre-compile CUDA kernels) ───────────────────────────────
if [[ "$DEVICE" == "cuda" ]]; then
    log "Warming up CUDA (compiling kernels)..."
    python3 -c "
import torch
# Pre-allocate and run a small tensor op to trigger JIT compilation
x = torch.randn(256, 256, device='cuda')
_ = torch.mm(x, x)
torch.cuda.synchronize()
print('  CUDA warm-up complete')
" 2>/dev/null && ok "CUDA kernels pre-compiled" || warn "CUDA warm-up skipped"
fi

# ── 9. Launch training ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════ TRAINING START ════════════════════${NC}"
if [[ "$DEVICE" == "cuda" ]]; then
    echo -e "  GPU: ${GREEN}${GPU_NAME}${NC} (${GPU_MEM_GB} GB) | Device: ${GREEN}${DEVICE}${NC}"
elif [[ "$DEVICE" == "mps" ]]; then
    echo -e "  Device: ${GREEN}Apple MPS${NC}"
else
    echo -e "  Device: ${YELLOW}CPU${NC}"
fi
echo ""

START_TIME=$(date +%s)

# VRAM_LIMIT may be set in CUDA block above; default to 7.0 if not
VRAM_LIMIT="${VRAM_LIMIT:-7.0}"

python3 "$PROJECT_ROOT/train.py" \
    --steps "$STEPS" \
    --episode-length "$EPISODE_LEN" \
    --eval-freq "$EVAL_FREQ" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --vram-limit "$VRAM_LIMIT" \
    $BATCH_ARGS \
    2>&1 | tee "$TRAIN_LOG"

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo ""
echo -e "${BOLD}════════════════════ TRAINING DONE ═════════════════════${NC}"
echo ""

if [[ $EXIT_CODE -eq 0 ]]; then
    ok "Training completed in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
else
    fail "Training failed with exit code $EXIT_CODE (see $TRAIN_LOG)"
fi

# ── 10. GPU memory summary ──────────────────────────────────────────────────
if [[ "$DEVICE" == "cuda" ]]; then
    python3 -c "
import torch
allocated = torch.cuda.max_memory_allocated() / (1024**3)
reserved  = torch.cuda.max_memory_reserved()  / (1024**3)
print(f'  Peak GPU memory: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved')
" 2>/dev/null || true
fi

# ── 11. Post-training analysis ──────────────────────────────────────────────
if [[ -f "$PROJECT_ROOT/analyze_training.py" ]]; then
    log "Running post-training analysis..."
    echo ""
    python3 "$PROJECT_ROOT/analyze_training.py" 2>&1 | tee -a "$TRAIN_LOG"
fi

# ── 12. Summary ─────────────────────────────────────────────────────────────
echo ""
BEST_COUNT=$(find "$LOG_DIR/best" -name "*.zip" 2>/dev/null | wc -l | tr -d ' ')
FINAL_COUNT=$(find "$LOG_DIR/final" -name "*.zip" 2>/dev/null | wc -l | tr -d ' ')

echo -e "${BOLD}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Training Complete                                       ║${NC}"
echo -e "${BOLD}╠═══════════════════════════════════════════════════════════╣${NC}"
printf "${BOLD}║${NC}   Duration:     %-41s${BOLD}║${NC}\n" "${ELAPSED_MIN}m ${ELAPSED_SEC}s"
printf "${BOLD}║${NC}   Steps:        %-41s${BOLD}║${NC}\n" "$STEPS"
printf "${BOLD}║${NC}   Device:       %-41s${BOLD}║${NC}\n" "$DEVICE ($GPU_NAME)"
printf "${BOLD}║${NC}   Best models:  %-41s${BOLD}║${NC}\n" "$BEST_COUNT agents saved"
printf "${BOLD}║${NC}   Final models: %-41s${BOLD}║${NC}\n" "$FINAL_COUNT agents saved"
printf "${BOLD}║${NC}   Log file:     %-41s${BOLD}║${NC}\n" "$(basename "$TRAIN_LOG")"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Models saved to: $LOG_DIR/{best,final}/"
log "Full log: $TRAIN_LOG"
