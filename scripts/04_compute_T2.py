#!/usr/bin/env python
"""Tier T2 anharmonic-uMLIP features (one uMLIP per invocation).

Computes mode-Grüneisen parameters from QHA (5 volumes) and the Knoop
anharmonicity score σ^A from 25 ps NVT at 300 K.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from umlip_kappa.features_T2 import compute_T2_for_material
from umlip_kappa.io_utils import ensure_dir, get_logger, load_config

log = get_logger()


def _worker(row, cfg, umlip, force):
    mp_id = row["mp_id"]
    structure_dict = json.loads(row["structure_dict"]) if isinstance(row["structure_dict"], str) else row["structure_dict"]
    return compute_T2_for_material(mp_id, structure_dict, umlip, cfg, force=force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--umlip", "-u", required=True)
    ap.add_argument("--workers", "-j", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    bench = pd.read_csv(cfg["paths"]["benchmark_csv"])
    ensure_dir(cfg["paths"]["output_dir"])

    rows = bench.to_dict("records")
    fn = partial(_worker, cfg=cfg, umlip=args.umlip, force=args.force)

    results = []
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(fn, rows), total=len(rows)):
                if r is not None:
                    results.append(r)
    else:
        for row in tqdm(rows):
            r = fn(row)
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    out = Path(cfg["paths"]["output_dir"]) / f"features_T2_{args.umlip}.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %d/%d T2 rows to %s", len(df), len(rows), out)


if __name__ == "__main__":
    main()
