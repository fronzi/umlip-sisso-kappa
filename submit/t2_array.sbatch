#!/bin/bash --login
# Tier-T2 anharmonic uMLIP features (QHA Grüneisen + Knoop σ^A).
# One material per array task. Wall-time is 3× T1 because of the NVT MD.
#SBATCH --job-name=ukappa-t2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --array=0-4%32
#SBATCH --output=logs/t2_%A_%a.out
#SBATCH --error=logs/t2_%A_%a.err

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


CFG=${UMLIP_KAPPA_CFG:-configs/setonix.yaml}
UMLIP=${UMLIP_KAPPA_UMLIP:-mace-mp-0}

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MP_ID=$(awk -F',' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n{print $1}' data/benchmark.csv)
[[ -n "$MP_ID" ]] || { echo "no mp_id for array idx $SLURM_ARRAY_TASK_ID"; exit 1; }

echo "[$(date)] T2 task $SLURM_ARRAY_TASK_ID  mp_id=$MP_ID  gpu=$ROCR_VISIBLE_DEVICES"

python - <<PYEOF
import json, pandas as pd
from umlip_kappa.io_utils import load_config
from umlip_kappa.features_T2 import compute_T2_for_material

cfg = load_config("$CFG")
df  = pd.read_csv(cfg["paths"]["benchmark_csv"])
row = df[df["mp_id"] == "$MP_ID"].iloc[0]
import ast
import json as _json
try:
    sd = _json.loads(row["structure_dict"]) if isinstance(row["structure_dict"], str) else row["structure_dict"]
except Exception:
    sd = ast.literal_eval(row["structure_dict"])
compute_T2_for_material("$MP_ID", sd, "$UMLIP", cfg, force=False)
PYEOF

echo "[$(date)] T2 task $SLURM_ARRAY_TASK_ID done"
