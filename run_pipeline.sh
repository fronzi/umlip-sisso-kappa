#!/bin/bash
# =============================================================================
# run_pipeline.sh — one-shot orchestrator for the umlip-sisso-kappa pipeline
#                   on Pawsey Setonix (or any SLURM cluster with minor edits).
#
# Submits, in dependency order:
#   1. cpu_stages.sbatch     — assemble dataset + Tier-T0 features (CPU)
#   2. t1_array.sbatch       — Tier-T1 harmonic uMLIP features (GPU array)
#   3. t2_array.sbatch       — Tier-T2 anharmonic uMLIP features (GPU array)
#   4. gather.sbatch         — collate per-material JSONs into CSVs (CPU)
#   5. cpu_stages.sbatch     — SISSO + softening + figures (CPU, second run)
#
# Behaviours:
#   * Refuses to submit if any required file is missing (preflight).
#   * Reads N (number of materials) from data/benchmark.csv if it exists,
#     otherwise from data/benchmark_seed.csv.
#   * Resizes the --array=0-(N-1) range automatically.
#   * Records every submitted job-ID in run.log and to a manifest file so
#     re-running skips stages whose final outputs already exist.
#   * Optional dry-run with -n; optional restart-from-stage with -s <1..5>.
#
# Usage:
#   ./run_pipeline.sh                       # full pipeline, default config
#   ./run_pipeline.sh -c configs/setonix.yaml -u mace-mp-0
#   ./run_pipeline.sh -n                    # dry-run: print sbatch commands
#   ./run_pipeline.sh -s 3                  # restart from stage 3 (T2 array)
#   ./run_pipeline.sh -A pawsey0123 -G pawsey0123-gpu
# =============================================================================

set -euo pipefail

# --------------------------- defaults (override with flags) ------------------
CONFIG="configs/setonix.yaml"
UMLIP="mace-mp-0"
ACCOUNT_CPU="${SBATCH_ACCOUNT_CPU:-}"          # e.g. pawsey0123
ACCOUNT_GPU="${SBATCH_ACCOUNT_GPU:-}"          # e.g. pawsey0123-gpu
ARRAY_CONCURRENCY=32                            # %N in --array
DRY_RUN=0
START_STAGE=1
END_STAGE=5

# --------------------------- argument parsing --------------------------------
usage() {
    sed -n 's/^# \{0,1\}//;1,/^$/p' "$0" | head -40
    exit 1
}
while getopts "c:u:A:G:k:s:e:nh" opt; do
    case $opt in
        c) CONFIG="$OPTARG" ;;
        u) UMLIP="$OPTARG" ;;
        A) ACCOUNT_CPU="$OPTARG" ;;
        G) ACCOUNT_GPU="$OPTARG" ;;
        k) ARRAY_CONCURRENCY="$OPTARG" ;;
        s) START_STAGE="$OPTARG" ;;
        e) END_STAGE="$OPTARG" ;;
        n) DRY_RUN=1 ;;
        h|*) usage ;;
    esac
done

# --------------------------- helpers -----------------------------------------
log()  { printf "[%(%F %T)T] %s\n" -1 "$*" | tee -a run.log; }
die()  { log "ERROR: $*"; exit 1; }
need() { [[ -e "$1" ]] || die "missing required file: $1"; }

submit() {
    # submit <stage_name> <sbatch_args...>
    # Stdout = the SLURM jobid (a single integer) so it can be captured with $(...)
    # Stderr = everything human-readable (also pulled into run.log by the caller).
    local name=$1; shift
    if (( DRY_RUN )); then
        echo "DRY-RUN: sbatch $* ($name)" >&2
        # Emit a fake but parsable-as-string jobid that does not contain spaces:
        echo "DRY_${name//[^A-Za-z0-9]/_}"
        return
    fi
    local out
    out=$(sbatch --parsable "$@") \
        || die "sbatch failed for $name (see error above)"
    log "submitted $name -> jobid $out"
    echo "$out"
}

# --------------------------- preflight ---------------------------------------
log "================================================================"
log "umlip-sisso-kappa pipeline launcher"
log "  config:           $CONFIG"
log "  uMLIP:            $UMLIP"
log "  CPU account:      ${ACCOUNT_CPU:-<must set with -A or SBATCH_ACCOUNT_CPU>}"
log "  GPU account:      ${ACCOUNT_GPU:-<must set with -G or SBATCH_ACCOUNT_GPU>}"
log "  array concurrency: %$ARRAY_CONCURRENCY"
log "  stages:           $START_STAGE .. $END_STAGE  (dry-run: $DRY_RUN)"
log "================================================================"

[[ -n "$ACCOUNT_CPU" && -n "$ACCOUNT_GPU" ]] \
    || die "set CPU and GPU accounts with -A and -G (or env vars)"

need "$CONFIG"
need "src/umlip_kappa/__init__.py"
need "scripts/01_assemble_dataset.py"
need "submit/cpu_stages.sbatch"
need "submit/t1_array.sbatch"
need "submit/t2_array.sbatch"
need "submit/gather.sbatch"
need "data/benchmark_seed.csv"
need "data/dft_reference_subset.csv"
need "$HOME/.config/mp_api_key"

command -v sbatch >/dev/null || { (( DRY_RUN )) && log "sbatch missing (dry-run, OK)" || die "sbatch not found — are you on a login node?"; }
command -v squeue >/dev/null || { (( DRY_RUN )) && log "squeue missing (dry-run, OK)" || die "squeue not found"; }

# Read N from the assembled benchmark CSV if it exists, else from the seed.
# If the assembled CSV does not exist yet, the seed gives an upper bound;
# stage 1 will resize the array script after assembly.
if [[ -f data/benchmark.csv ]]; then
    N=$(($(wc -l < data/benchmark.csv) - 1))
    log "found data/benchmark.csv with N=$N materials"
else
    N=$(($(wc -l < data/benchmark_seed.csv) - 1 - $(grep -c '^#' data/benchmark_seed.csv || true)))
    log "data/benchmark.csv not yet built; seed has N≈$N (will be re-checked after stage 1)"
fi
[[ "$N" -ge 1 ]] || die "N=$N — benchmark CSV is empty"

# Resize the SLURM array ranges in-place (only if needed)
resize_array() {
    local sbatch_file=$1
    local new_range="0-$((N-1))%${ARRAY_CONCURRENCY}"
    # Match `--array=...` and replace whatever is on the RHS until end-of-line
    sed -i.bak -E "s|^#SBATCH --array=.*|#SBATCH --array=${new_range}|" "$sbatch_file"
    log "resized array in $sbatch_file -> ${new_range}"
}
resize_array submit/t1_array.sbatch
resize_array submit/t2_array.sbatch

mkdir -p logs manifests
MANIFEST="manifests/run_$(date +%Y%m%d_%H%M%S).env"
log "manifest: $MANIFEST"

# --------------------------- already-done detection --------------------------
# Outputs we look at to decide whether a stage's job needs to be submitted.
# These paths must agree with paths.output_dir in the YAML config.
OUTPUT_DIR=$(python - <<EOF
from umlip_kappa.io_utils import load_config
print(load_config("$CONFIG")["paths"]["output_dir"])
EOF
)
log "output_dir from config: $OUTPUT_DIR"

stage_already_done() {
    case $1 in
        1) [[ -f data/benchmark.csv && -f "$OUTPUT_DIR/features_T0.csv" ]] ;;
        2) [[ -f "$OUTPUT_DIR/features_T1_${UMLIP}.csv" ]] \
              || { count_done=$(find "$OUTPUT_DIR" -maxdepth 2 -name "T1_${UMLIP}.json" 2>/dev/null | wc -l)
                   [[ "$count_done" -ge "$N" ]]; } ;;
        3) [[ -f "$OUTPUT_DIR/features_T2_${UMLIP}.csv" ]] \
              || { count_done=$(find "$OUTPUT_DIR" -maxdepth 2 -name "T2_${UMLIP}.json" 2>/dev/null | wc -l)
                   [[ "$count_done" -ge "$N" ]]; } ;;
        4) [[ -f "$OUTPUT_DIR/features_T1_${UMLIP}.csv" && -f "$OUTPUT_DIR/features_T2_${UMLIP}.csv" ]] ;;
        5) [[ -f "$OUTPUT_DIR/sr/pareto_T2_${UMLIP}.json" ]] ;;
        *) false ;;
    esac
}

# --------------------------- submit chain ------------------------------------
declare -A JOBID

for stage in 1 2 3 4 5; do
    (( stage < START_STAGE )) && continue
    (( stage > END_STAGE )) && break

    if stage_already_done "$stage"; then
        log "stage $stage: outputs already present, skipping"
        continue
    fi

    # Build the --dependency string from the previous stage's jobid, if any.
    DEP=""
    prev=$((stage - 1))
    if (( prev >= START_STAGE )) && [[ -n "${JOBID[$prev]:-}" ]]; then
        DEP="--dependency=afterany:${JOBID[$prev]}"
    fi

    case $stage in
        1)  # dataset + T0 (CPU)
            JOBID[1]=$(submit "stage1-cpu-dataset+T0" \
                $DEP \
                --account="$ACCOUNT_CPU" \
                --export=ALL,UMLIP_KAPPA_CFG="$CONFIG",UMLIP_KAPPA_STAGE="dataset_and_t0" \
                submit/cpu_stages.sbatch)
            ;;
        2)  # T1 GPU array
            JOBID[2]=$(submit "stage2-gpu-T1-array" \
                $DEP \
                --account="$ACCOUNT_GPU" \
                --export=ALL,UMLIP_KAPPA_CFG="$CONFIG",UMLIP_KAPPA_UMLIP="$UMLIP" \
                submit/t1_array.sbatch)
            ;;
        3)  # T2 GPU array
            JOBID[3]=$(submit "stage3-gpu-T2-array" \
                $DEP \
                --account="$ACCOUNT_GPU" \
                --export=ALL,UMLIP_KAPPA_CFG="$CONFIG",UMLIP_KAPPA_UMLIP="$UMLIP" \
                submit/t2_array.sbatch)
            ;;
        4)  # gather per-material JSONs -> CSVs (CPU)
            JOBID[4]=$(submit "stage4-cpu-gather" \
                $DEP \
                --account="$ACCOUNT_CPU" \
                --export=ALL,UMLIP_KAPPA_CFG="$CONFIG",UMLIP_KAPPA_UMLIP="$UMLIP" \
                submit/gather.sbatch)
            ;;
        5)  # SISSO + softening + figures (CPU)
            JOBID[5]=$(submit "stage5-cpu-sisso+plots" \
                $DEP \
                --account="$ACCOUNT_CPU" \
                --export=ALL,UMLIP_KAPPA_CFG="$CONFIG",UMLIP_KAPPA_UMLIP="$UMLIP",UMLIP_KAPPA_STAGE="sisso_and_plots" \
                submit/cpu_stages.sbatch)
            ;;
    esac
done

# --------------------------- write manifest ---------------------------------
{
    echo "# pipeline manifest $(date -u +%FT%TZ)"
    echo "CONFIG=$CONFIG"
    echo "UMLIP=$UMLIP"
    echo "N=$N"
    echo "ARRAY_CONCURRENCY=$ARRAY_CONCURRENCY"
    for k in "${!JOBID[@]}"; do echo "JOBID_$k=${JOBID[$k]}"; done
} > "$MANIFEST"
log "wrote $MANIFEST"

# --------------------------- summary -----------------------------------------
log "================================================================"
log "submitted job-IDs (in dependency order):"
for stage in $(echo "${!JOBID[@]}" | tr ' ' '\n' | sort); do
    log "  stage $stage  ->  ${JOBID[$stage]}"
done
log ""
log "monitor with:"
log "  squeue --me"
log "  sacct -j ${JOBID[${START_STAGE}]:-<first-jobid>} --format=JobID,State,Elapsed,MaxRSS"
log "  tail -F logs/t1_${JOBID[2]:-<jobid>}_0.out"
log ""
log "to resume after a partial failure, just re-run this script — checkpointed"
log "stages will be skipped automatically."
log "================================================================"
