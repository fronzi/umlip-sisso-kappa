#!/usr/bin/env python
"""Quantify how PES softening propagates into the discovered descriptor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from umlip_kappa.io_utils import ensure_dir, get_logger, load_config
from umlip_kappa.validate import softening_decomposition, parity_metrics

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--tier", default="T2")
    ap.add_argument("--umlip", "-u", required=True)
    ap.add_argument("--top-k", type=int, default=1,
                    help="Use the top-K Pareto-optimal models.")
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["paths"]["output_dir"]) / "sr"
    pf = json.loads((out_dir / f"pareto_{args.tier}_{args.umlip}.json").read_text())
    data = pd.read_csv(out_dir / f"data_{args.tier}_{args.umlip}.csv").set_index("mp_id")

    ref_csv = cfg["validation"]["dft_reference_subset_csv"]
    dft = pd.read_csv(ref_csv).set_index("mp_id")
    log.info("DFT reference subset: %d materials", len(dft))

    results = []
    for entry in pf[: args.top_k]:
        formula = entry["formula"]
        feature_names = [c for c in dft.columns if c in data.columns]
        decomp = softening_decomposition(formula, feature_names, data, dft)
        decomp.to_csv(out_dir / f"softening_{args.tier}_{args.umlip}_c{entry['complexity']}.csv",
                      index=False)
        log.info("Softening decomposition for complexity %d -> top feature: %s",
                 entry["complexity"], decomp.iloc[0]["feature"])
        results.append({"formula": formula, "decomp": decomp.to_dict("records")})

    (out_dir / f"softening_summary_{args.tier}_{args.umlip}.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
