#!/bin/bash --login
# Collate per-material T1_*.json and T2_*.json checkpoints into wide CSVs.
# Runs after both GPU arrays complete; cheap CPU job.
#SBATCH --job-name=ukappa-gather
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gather_%j.out
#SBATCH --error=logs/gather_%j.err

set -euo pipefail

# --- Setonix env (does not auto-propagate into SLURM jobs) ---
: "${MYSOFTWARE:=/software/projects/pawsey0328/$USER}"
export MYSOFTWARE
export PATH="$MYSOFTWARE/bin:$PATH"

module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate umlip-kappa

CFG=${UMLIP_KAPPA_CFG:-configs/setonix.yaml}
UMLIP=${UMLIP_KAPPA_UMLIP:-mace-mp-0}

cd "$SLURM_SUBMIT_DIR"

python - <<PYEOF
import json, glob, pandas as pd, pathlib
from umlip_kappa.io_utils import load_config

cfg = load_config("$CFG")
out = pathlib.Path(cfg["paths"]["output_dir"])

for tier in ("T1", "T2"):
    rows, n_err = [], 0
    for fp in sorted(out.glob(f"*/{tier}_${UMLIP}.json")):
        try:
            rows.append(json.load(open(fp)))
        except Exception as e:
            print(f"  WARN  could not read {fp}: {e}")
            n_err += 1
    if not rows:
        print(f"{tier}: 0 rows — no per-material checkpoints found")
        continue
    df = pd.DataFrame(rows)
    csv = out / f"features_{tier}_${UMLIP}.csv"
    df.to_csv(csv, index=False)
    print(f"{tier}: {len(df)} rows ({n_err} errors)  ->  {csv}")
PYEOF

echo "[$(date)] gather done"
