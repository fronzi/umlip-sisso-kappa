#!/bin/bash --login
# Tier-T1 harmonic uMLIP features. One material per array task.
#SBATCH --job-name=ukappa-t1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-4%32
#SBATCH --output=logs/t1_%A_%a.out
#SBATCH --error=logs/t1_%A_%a.err

set -euo pipefail

# --- Setonix env (does not auto-propagate into SLURM jobs) ---
: "${MYSOFTWARE:=/software/projects/pawsey0328/$USER}"
export MYSOFTWARE
export PATH="$MYSOFTWARE/bin:$PATH"

module purge
# module load rocm — already loaded by pawseyenv/2025.08
eval "$(micromamba shell hook --shell bash)"
micromamba activate umlip-kappa
source "$HOME/.config/mp_api_key"

# Pin to the GCD that SLURM assigned (Setonix MI250X = 1 GCD per --gres=gpu:1)

CFG=${UMLIP_KAPPA_CFG:-configs/setonix.yaml}
UMLIP=${UMLIP_KAPPA_UMLIP:-mace-mp-0}

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Pull the mp_id for this array task from data/benchmark.csv (skip header)
MP_ID=$(awk -F',' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n{print $1}' data/benchmark.csv)
[[ -n "$MP_ID" ]] || { echo "no mp_id for array idx $SLURM_ARRAY_TASK_ID"; exit 1; }

echo "[$(date)] T1 task $SLURM_ARRAY_TASK_ID  mp_id=$MP_ID  gpu=$ROCR_VISIBLE_DEVICES"

python - <<PYEOF
import json, pandas as pd
from umlip_kappa.io_utils import load_config
from umlip_kappa.features_T1 import compute_T1_for_material

cfg = load_config("$CFG")
df  = pd.read_csv(cfg["paths"]["benchmark_csv"])
row = df[df["mp_id"] == "$MP_ID"].iloc[0]
import ast
import json as _json
try:
    sd = _json.loads(row["structure_dict"]) if isinstance(row["structure_dict"], str) else row["structure_dict"]
except Exception:
    sd = ast.literal_eval(row["structure_dict"])
compute_T1_for_material("$MP_ID", sd, "$UMLIP", cfg, force=False)
PYEOF

echo "[$(date)] T1 task $SLURM_ARRAY_TASK_ID done"
