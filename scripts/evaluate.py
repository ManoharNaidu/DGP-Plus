"""Print a results table for one or more configs: mean ± std over seeds.

    python scripts/evaluate.py configs/runs/*.yaml
"""

import argparse

import _common  # noqa: F401  (puts the package on the path)

from dgp_plus.config import load_config
from dgp_plus.evaluate import METRICS
from dgp_plus.runs import summarise_runs


def main():
    made = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    made.add_argument("configs", nargs="+")
    args = made.parse_args()
    print("| Config | Seeds | Macro-F1 | AUROC | AUPRC |")
    print("|---|---|---|---|---|")
    for path in args.configs:
        summary = summarise_runs(load_config(path))
        cells = [f"{summary['test'][m]['mean']:.2f} ± {summary['test'][m]['std']:.2f}" for m in METRICS] if summary else ["no runs"] * 3
        seeds = len(summary["seeds"]) if summary else 0
        print(f"| {path} | {seeds} | {' | '.join(cells)} |")


if __name__ == "__main__":
    main()
