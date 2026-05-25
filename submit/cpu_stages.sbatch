#!/bin/bash --login
# Run on the CPU partition, used for two stages (controlled by env var):
#   UMLIP_KAPPA_STAGE=dataset_and_t0   -> assemble dataset + Tier T0
#   UMLIP_KAPPA_STAGE=sisso_and_plots  -> SISSO + softening + figures
#SBATCH --job-name=ukappa-cpu
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cpu_%x_%j.out
#SBATCH --error=logs/cpu_%x_%j.err

set -euo pipefail

# --- Setonix env (does not auto-propagate into SLURM jobs) ---
: "${MYSOFTWARE:=/software/projects/pawsey0328/$USER}"
export MYSOFTWARE
export PATH="$MYSOFTWARE/bin:$PATH"

module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate umlip-kappa
source "$HOME/.config/mp_api_key"

CFG=${UMLIP_KAPPA_CFG:-configs/setonix.yaml}
UMLIP=${UMLIP_KAPPA_UMLIP:-mace-mp-0}
STAGE=${UMLIP_KAPPA_STAGE:-dataset_and_t0}

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[$(date)] CPU stage=$STAGE  cfg=$CFG  umlip=$UMLIP"

case "$STAGE" in
    dataset_and_t0)
        python scripts/01_assemble_dataset.py -c "$CFG"
        python scripts/02_compute_T0.py       -c "$CFG"
        ;;
    sisso_and_plots)
        for tier in T0 T1 T2; do
            python scripts/05_run_sisso.py    -c "$CFG" -t "$tier" -u "$UMLIP"
        done
        python scripts/06_validate_softening.py -c "$CFG" -u "$UMLIP"
        python scripts/07_make_figures.py       -c "$CFG" -u "$UMLIP"
        ;;
    *)
        echo "unknown UMLIP_KAPPA_STAGE='$STAGE'" >&2
        exit 1
        ;;
esac

echo "[$(date)] CPU stage=$STAGE done"
