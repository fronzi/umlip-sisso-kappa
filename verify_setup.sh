#!/bin/bash
# verify_setup.sh — preflight sanity checks for the Setonix pipeline.
# Run this on the login node BEFORE invoking run_pipeline.sh. Catches missing
# files, wrong Python env, account-name typos, and most "obvious" config bugs.

set -uo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}!${NC} %s\n" "$*"; ((WARN++)); }
fail() { printf "${RED}✗${NC} %s\n" "$*"; ((FAIL++)); }

WARN=0; FAIL=0
CFG=${1:-configs/setonix.yaml}

echo "umlip-sisso-kappa preflight, using config: $CFG"
echo "----------------------------------------------------------------"

# ---------- files ----------
for f in pyproject.toml configs/setonix.yaml src/umlip_kappa/__init__.py \
         scripts/01_assemble_dataset.py scripts/03_compute_T1.py \
         scripts/04_compute_T2.py scripts/05_run_sisso.py \
         submit/cpu_stages.sbatch submit/t1_array.sbatch \
         submit/t2_array.sbatch submit/gather.sbatch \
         data/benchmark_seed.csv data/dft_reference_subset.csv ; do
    [[ -f "$f" ]] && ok "found $f" || fail "missing $f"
done

[[ -f "$HOME/.config/mp_api_key" ]] \
    && ok "found ~/.config/mp_api_key" \
    || fail "missing ~/.config/mp_api_key — see README §Installation step 5"

# ---------- conda / python ----------
if command -v mamba >/dev/null; then ok "mamba on PATH"; else fail "mamba not found"; fi
if [[ "${CONDA_DEFAULT_ENV:-}" == "umlip-kappa" ]]; then
    ok "conda env umlip-kappa is active"
else
    warn "conda env not active — activate with 'mamba activate umlip-kappa'"
fi

# ---------- python imports ----------
if python -c "import umlip_kappa" 2>/dev/null; then
    ok "umlip_kappa importable"
else
    fail "umlip_kappa not importable — did you 'pip install -e .'?"
fi
for mod in ase phonopy pymatgen matminer numpy scipy pandas sklearn yaml; do
    if python -c "import $mod" 2>/dev/null; then ok "import $mod"; else fail "import $mod"; fi
done

# ---------- config readable + paths sensible ----------
python - <<PYEOF
import sys, pathlib
from umlip_kappa.io_utils import load_config
cfg = load_config("$CFG")
out = pathlib.Path(cfg["paths"]["output_dir"])
if "<project>" in str(out) or "<user>" in str(out):
    print(f"FAIL_CFG output_dir still has placeholders: {out}")
    sys.exit(1)
out.parent.mkdir(parents=True, exist_ok=True)
if not out.parent.exists():
    print(f"FAIL_CFG cannot create {out.parent}")
    sys.exit(1)
# Confirm it's on /scratch on Setonix (warn otherwise)
if not str(out).startswith("/scratch"):
    print(f"WARN_CFG output_dir not on /scratch — fine on a laptop but check Setonix")
print("CFG_OK")
PYEOF
case $? in
    0) ok "config parses and paths are usable" ;;
    *) fail "config check failed (see message above)" ;;
esac

# ---------- SLURM ----------
if command -v sbatch >/dev/null; then ok "sbatch available"; else fail "sbatch not on PATH"; fi
if command -v squeue >/dev/null; then ok "squeue available"; else fail "squeue not on PATH"; fi

# ---------- accounts (just informational) ----------
if [[ -n "${SBATCH_ACCOUNT_CPU:-}" ]]; then
    ok "SBATCH_ACCOUNT_CPU=$SBATCH_ACCOUNT_CPU"
else
    warn "SBATCH_ACCOUNT_CPU not set — pass with run_pipeline.sh -A <project>"
fi
if [[ -n "${SBATCH_ACCOUNT_GPU:-}" ]]; then
    ok "SBATCH_ACCOUNT_GPU=$SBATCH_ACCOUNT_GPU"
else
    warn "SBATCH_ACCOUNT_GPU not set — pass with run_pipeline.sh -G <project>-gpu"
fi

# ---------- MACE model cache ----------
if [[ -d "$HOME/.cache/mace" ]] && find "$HOME/.cache/mace" -name '*.model' | grep -q .; then
    ok "MACE checkpoint cache populated"
else
    warn "MACE checkpoint not cached — on the login node run once:"
    warn "    python -c \"from mace.calculators import mace_mp; mace_mp(model='medium', device='cpu')\""
fi

# ---------- benchmark sanity ----------
N_seed=$(grep -vc '^#' data/benchmark_seed.csv 2>/dev/null || echo 0)
N_seed=$((N_seed - 1))                 # subtract header
if [[ "$N_seed" -lt 5 ]]; then
    warn "seed CSV has only $N_seed materials — too small for a real run"
else
    ok "seed CSV has $N_seed materials"
fi

echo "----------------------------------------------------------------"
printf "result: %d ok, %d warnings, %d failures\n" \
    "$(( $(grep -c '✓' < <(true)) ))" "$WARN" "$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
    echo "FIX the failures above before submitting."
    exit 1
fi
[[ "$WARN" -gt 0 ]] && echo "Address the warnings, then run ./run_pipeline.sh"
exit 0
