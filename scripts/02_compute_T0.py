#!/usr/bin/env python
"""Tier T0 (composition + density) features. Fast, no uMLIP, no DFT."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from umlip_kappa.features_T0 import compute_T0_features
from umlip_kappa.io_utils import ensure_dir, get_logger, load_config

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    bench = pd.read_csv(cfg["paths"]["benchmark_csv"])
    import ast
    def _parse(s):
        if not isinstance(s, str):
            return s
        try:
            return json.loads(s)
        except Exception:
            return ast.literal_eval(s)   # legacy CSV with Python-repr dicts
    bench["structure_dict"] = bench["structure_dict"].apply(_parse)

    feats = compute_T0_features(bench)
    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    out_path = out_dir / "features_T0.csv"
    feats.to_csv(out_path, index=False)
    log.info("Wrote T0 features (%d rows × %d cols) to %s",
             feats.shape[0], feats.shape[1], out_path)


if __name__ == "__main__":
    main()
