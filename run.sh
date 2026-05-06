#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run.sh  ·  MekongTunnel video generator control script
#
#  USAGE:
#    ./run.sh run                   Full run, all 3 GPUs, parallel
#    ./run.sh run --gpus 1          Use only GPU 0
#    ./run.sh run --gpus 2          Use GPU 0+1
#    ./run.sh run --quiet           Low power: 10 steps, 150W limit, sequential
#    ./run.sh run --gpus 2 --quiet  2 GPUs, low power
#    ./run.sh test                  Quick test: scene_01 only, 1 GPU, 5 steps
#    ./run.sh stitch                Assemble clips → final MP4
#    ./run.sh dry                   Show scene plan, no generation
#    ./run.sh status                GPU temps, memory, utilisation
#    ./run.sh kill                  Kill all generation processes + free VRAM
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="./venv/bin/python"
PIP="./venv/bin/pip"
LOGFILE="logs/generate.log"
PIDFILE="/tmp/vdo_gen.pid"

# ── Defaults ─────────────────────────────────────────────────────────────────
GPU_COUNT=3       # number of GPUs to use (1 / 2 / 3)
QUIET=false       # low-power mode
STEPS_NORMAL=25   # inference steps for full quality
STEPS_QUIET=10    # inference steps for quiet mode
POWER_NORMAL=400  # watts per GPU (normal)
POWER_QUIET=150   # watts per GPU (quiet — fans barely spin)

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}▶ $*${RESET}"; }
success() { echo -e "${GREEN}✓ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
error()   { echo -e "${RED}✗ $*${RESET}" >&2; }
header()  { echo -e "\n${BOLD}━━━ $* ━━━${RESET}\n"; }

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

gpu_ids_arg() {
    # Build comma-separated GPU IDs from GPU_COUNT
    case $GPU_COUNT in
        1) echo "0" ;;
        2) echo "0,1" ;;
        3) echo "0,1,2" ;;
        *) echo "0" ;;
    esac
}

set_power_limit() {
    local watts=$1
    info "Setting GPU power limit to ${watts}W per card..."
    for gpu in $(seq 0 $((GPU_COUNT - 1))); do
        if nvidia-smi -i "$gpu" -pl "$watts" &>/dev/null; then
            success "GPU $gpu → ${watts}W"
        else
            warn "Could not set power limit on GPU $gpu (may need root)"
        fi
    done
}

restore_power_limit() {
    info "Restoring GPU power limits to ${POWER_NORMAL}W..."
    for gpu in $(seq 0 $((GPU_COUNT - 1))); do
        nvidia-smi -i "$gpu" -pl "$POWER_NORMAL" &>/dev/null || true
    done
}

kill_all() {
    header "KILL ALL"
    # Kill by PID file
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            info "Killing process group $pid..."
            kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
            success "Killed PID $pid"
        fi
        rm -f "$PIDFILE"
    fi
    # Kill any stray python generation processes
    pkill -f "generate.py"   2>/dev/null && success "Killed generate.py processes" || true
    pkill -f "stitch_video"  2>/dev/null && success "Killed stitch_video processes" || true
    # Restore power to default
    restore_power_limit
    success "All clear. GPUs free."
}

show_status() {
    header "GPU STATUS"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
               --format=csv,noheader,nounits | \
    awk -F', ' '{
        printf "  GPU %s  %-26s  Temp: %3s°C  Util: %3s%%  VRAM: %5s/%5s MiB  Power: %6s/%sW\n",
               $1,$2,$3,$4,$5,$6,$7,$8
    }'
    echo ""
    local procs
    procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
                       --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$procs" ]]; then
        warn "Running GPU processes:"
        echo "$procs" | awk -F', ' '{printf "    PID %-8s  %-40s  %s MiB\n",$1,$2,$3}'
    else
        success "No GPU processes running"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Generation runner
# ─────────────────────────────────────────────────────────────────────────────

do_run() {
    local scene_filter="${1:-}"   # empty = all scenes
    local steps="${2:-}"          # empty = use mode default
    local mode_label="${3:-FULL}"

    header "GENERATE  [$mode_label]  GPUs=$(gpu_ids_arg)  Steps=${steps:-default}"

    mkdir -p logs output

    # Set power limits before starting
    if $QUIET; then
        set_power_limit "$POWER_QUIET"
    else
        set_power_limit "$POWER_NORMAL"
    fi

    # Build generate.py arguments
    local args=()
    args+=("--gpus" "$(gpu_ids_arg)")
    [[ -n "$steps"        ]] && args+=("--steps" "$steps")
    [[ -n "$scene_filter" ]] && args+=("--scenes" "$scene_filter")
    $QUIET && args+=("--sequential") || args+=("--parallel")

    info "Running: $PYTHON generate.py ${args[*]}"
    info "Logs:    tail -f $LOGFILE"
    echo ""

    # Run in process group so kill_all can reach all workers
    set +e
    setsid "$PYTHON" generate.py "${args[@]}" &
    local gen_pid=$!
    echo "$gen_pid" > "$PIDFILE"
    wait "$gen_pid"
    local exit_code=$?
    rm -f "$PIDFILE"
    set -e

    restore_power_limit

    if [[ $exit_code -eq 0 ]]; then
        success "Generation complete!  Clips saved to output/"
        echo ""
        ls -lh output/*.mp4 2>/dev/null || true
    else
        error "Generation failed (exit $exit_code). Check: tail -50 $LOGFILE"
        exit $exit_code
    fi
}

do_stitch() {
    header "STITCH"
    info "Assembling clips → output/mekongtunnel_reel.mp4"
    "$PYTHON" stitch_video.py
    success "Done!"
    ls -lh output/mekongtunnel_reel.mp4 2>/dev/null || true
}

# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────────────────────

COMMAND="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_COUNT="$2"; shift 2 ;;
        --quiet)
            QUIET=true; shift ;;
        *)
            warn "Unknown option: $1"; shift ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

case "$COMMAND" in

    run)
        if $QUIET; then
            do_run "" "$STEPS_QUIET" "QUIET · low power"
        else
            do_run "" "$STEPS_NORMAL" "FULL QUALITY"
        fi
        ;;

    test)
        # Quick single-scene smoke test — 1 GPU, 5 steps
        GPU_COUNT=1
        do_run "scene_01" "5" "TEST · scene_01 · 1 GPU · 5 steps"
        ;;

    stitch)
        do_stitch
        ;;

    dry)
        header "DRY RUN"
        "$PYTHON" generate.py --dry-run
        ;;

    status)
        show_status
        ;;

    kill)
        kill_all
        ;;

    logs)
        tail -f "$LOGFILE"
        ;;

    help|--help|-h|"")
        echo ""
        echo -e "${BOLD}  MekongTunnel Video Generator${RESET}"
        echo ""
        echo "  COMMANDS:"
        echo "    run                  Full quality run (all GPUs, ${STEPS_NORMAL} steps)"
        echo "    run --quiet          Quiet mode: ${STEPS_QUIET} steps, ${POWER_QUIET}W/GPU, sequential"
        echo "    run --gpus 1         Single GPU run"
        echo "    run --gpus 2         Two GPUs"
        echo "    run --gpus 2 --quiet Two GPUs, quiet mode"
        echo "    test                 Quick test: scene_01, 1 GPU, 5 steps"
        echo "    stitch               Assemble clips → final MP4"
        echo "    dry                  Show scene plan (no generation)"
        echo "    status               GPU temps, memory, power"
        echo "    logs                 Tail the generation log"
        echo "    kill                 Kill all processes + restore GPU power"
        echo ""
        echo "  FAN NOISE GUIDE:"
        echo "    Full quality  →  ${POWER_NORMAL}W/GPU  → fans loud"
        echo "    --quiet       →  ${POWER_QUIET}W/GPU  → fans barely spin (~35°C idle)"
        echo "    --gpus 1      →  only one card spins up"
        echo ""
        ;;

    *)
        error "Unknown command: $COMMAND"
        echo "Run ./run.sh help for usage."
        exit 1
        ;;
esac
