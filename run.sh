#!/bin/bash
# AlphaGPT — Pipeline Runner
# Usage: bash run.sh {build|train|all}

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source myenv/bin/activate

mkdir -p logs

run_step() {
    local script="$1"   # Python file to run
    local label="$2"    # display name for log file & progress

    local timestamp=$(date +%Y%m%d%H%M%S)
    local log_file="./logs/${label}_${timestamp}.log"

    echo "=================================================="
    echo "  Step:   ${label}"
    echo "  Script: ${script}"
    echo "  Start:  $(date)"
    echo "  Log:    ${log_file}"
    echo "=================================================="

    python -u "${script}" 2>&1 | tee "${log_file}"

    echo "=================================================="
    echo "  Done:   ${label}"
    echo "  End:    $(date)"
    echo "=================================================="
    echo
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    build)
        run_step z_build_dataset.py "01_build_dataset"
        ;;
    train)
        run_step train_lgbm.py "02_train_lgbm"
        ;;
    optimize)
        run_step z_portfolio_optimize.py "03_portfolio_optimize"
        ;;
    *)
        echo "Usage: bash run.sh {build|train|optimize}"
        echo ""
        echo "  build      Rebuild the alpha-factor dataset"
        echo "  train      Train LightGBM model"
        echo "  optimize   Run portfolio optimization"
        exit 1
        ;;
esac
