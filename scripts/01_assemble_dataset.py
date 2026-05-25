#!/usr/bin/env python
"""Assemble the benchmark dataset (MP structures + filters + stratified split)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from umlip_kappa.dataset import assemble_benchmark
from umlip_kappa.io_utils import ensure_dir, get_logger, load_config

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--seed-csv", default=None,
                    help="Override the seed CSV path from the config.")
    args = ap.parse_args()
    cfg = load_config(args.config)

    seed = args.seed_csv or cfg["paths"]["benchmark_seed_csv"]
    if not Path(seed).exists():
        raise FileNotFoundError(
            f"Seed CSV not found at {seed}. See README §Dataset for the expected schema.")

    ensure_dir(cfg["paths"]["data_dir"])
    df = assemble_benchmark(seed, cfg, api_key=os.environ.get("MP_API_KEY"))
    out = cfg["paths"]["benchmark_csv"]
    df.to_csv(out, index=False)
    log.info("Wrote %d materials to %s", len(df), out)


if __name__ == "__main__":
    main()
