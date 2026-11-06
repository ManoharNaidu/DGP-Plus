"""Compare DGP+ with a baseline per dataset, and K = 1 with K = 2. Writes docs/DGP_PLUS_RESULTS.md and docs/dgp_plus_results.json.

    python scripts/compare_arms.py --baseline baseline/
    python scripts/compare_arms.py --baseline baseline/ --datasets instagram reddit

The baseline folder holds <dataset>/seed<s>/metrics.json and predictions.csv from any runs on the same test nodes.

Each comparison is paired over seeds 0-4: mean difference, seeds better, paired t, Holm over all tests of one dataset,
and a paired bootstrap 95% CI over test nodes, with each run's validation threshold held fixed for Macro-F1. The t-test
treats seeds as replicates, and the bootstrap covers test-set sampling.
"""

import argparse
import csv
import json
from pathlib import Path

import _common  # noqa: F401  (puts the package on the path)
import numpy as np

from dgp_plus.config import config_hash, load_config
from dgp_plus.evaluate import MIN_PAIRED_SEEDS, METRICS, aggregate, holm, paired_bootstrap, paired_test

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(5)
LABEL = {"macro_f1": "Macro-F1", "auroc": "AUROC", "auprc": "AUPRC"}

DATASETS = ("amazonvideo", "yelpchi", "instagram", "reddit", "amazon_musical")


def read_scores(folder: Path, name: str = "predictions.csv") -> tuple[np.ndarray, np.ndarray]:
    """Test labels and scores, sorted by node so that all arms line up."""
    rows = [r for r in csv.DictReader(open(folder / name, encoding="utf-8")) if r["split"] == "test"]
    rows.sort(key=lambda r: int(r["node"]))
    return np.array([int(r["label"]) for r in rows]), np.array([float(r["p_fraud"]) for r in rows])


def load_arm(pattern: str, rule: str | None = None) -> dict:
    """Seed -> (metrics, labels, scores), at the rule's epoch when one is named."""
    arm = {}
    for seed in SEEDS:
        folder = Path(pattern.format(s=seed))
        if not (folder / "metrics.json").exists():
            continue
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        file = "predictions.csv"
        if rule:
            kept = metrics["selection"][rule]
            if kept["epoch"] != metrics["best_epoch"]:
                file = f"predictions_epoch{kept['epoch']}.csv"
            metrics = {**metrics, "test": kept["test"], "best_epoch": kept["epoch"]}
        arm[seed] = (metrics, *read_scores(folder, file))
    return arm


def plus_arm(config: Path, rule: str | None = None) -> dict:
    cfg = load_config(config)
    return load_arm(str(ROOT / cfg["paths"]["results"] / "raw" / f"{cfg['dataset']['name']}_seed{{s}}_{config_hash(cfg)}"), rule)


def compare(arm: dict, reference: dict, metric: str) -> dict:
    return paired_test({s: run[0]["test"][metric] for s, run in arm.items()},
                       {s: run[0]["test"][metric] for s, run in reference.items()})


def row(arm: dict) -> dict:
    return aggregate([arm[s][0]["test"] for s in sorted(arm)])


def cell(summary: dict) -> str:
    return " | ".join(f"{summary[m]['mean']:.2f} ± {summary[m]['std']:.2f}" for m in METRICS)


def main():
    made = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    made.add_argument("--baseline", type=Path, required=True)
    made.add_argument("--datasets", nargs="+", default=DATASETS)
    args = made.parse_args()
    lines = ["# DGP+ results", "", "Test set, mean ± population std over seeds 0-4. Epoch and threshold chosen on validation only.",
             "Δ = arm − reference, mean over paired seeds. CI = paired bootstrap 95% over test nodes, 2,000 rounds.",
             "p = paired t over seeds. Holm is applied over all tests of one dataset.",
             f"A comparison with fewer than {MIN_PAIRED_SEEDS} paired seeds is shown as n/a.", ""]
    output = {}
    for dataset in args.datasets:
        plus = {f"DGP+ K={k}": plus_arm(ROOT / "configs" / "runs" / f"{dataset}_k{k}.yaml") for k in (1, 2)}
        tracked = {f"DGP+ K=2 {rule}": plus_arm(ROOT / "configs" / "runs" / f"{dataset}_k2.yaml", rule)
                   for rule in ("loss_es2", "auroc_all")}
        baseline = load_arm(str(args.baseline / dataset / "seed{s}"))
        lines += [f"## {dataset}", "", "| Arm | Seeds | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |", "|---|---|---|---|---|---|",
                  f"| DGP reproduced | {len(baseline)} | {cell(row(baseline))} | |"]
        for name, arm in {**plus, **tracked}.items():
            if not arm:
                lines.append(f"| {name} | 0 | not finished | | | |")
                continue
            seed = min(arm)
            k = 1 if "K=1" in name else 2
            config = load_config(ROOT / "configs" / "runs" / f"{dataset}_k{k}.yaml")
            manifest = ROOT / config["paths"]["results"] / "raw" / f"{config['dataset']['name']}_seed{seed}_{config_hash(config)}" / "run_manifest.json"
            tokens = json.loads(manifest.read_text(encoding="utf-8"))["prompt_tokens"]["mean"]
            lines.append(f"| {name} | {len(arm)} | {cell(row(arm))} | {tokens:.1f} |")
        tests, p_values = [], {}
        for name, arm in plus.items():
            if not arm:
                continue
            for reference_name, reference, metrics in [("DGP reproduced", baseline, METRICS),
                                                       *([("DGP+ K=2", plus["DGP+ K=2"], METRICS)] if name == "DGP+ K=1" and plus["DGP+ K=2"] else [])]:
                seeds = sorted(set(arm) & set(reference))
                ci = {}
                if len(seeds) >= MIN_PAIRED_SEEDS:
                    labels = arm[seeds[0]][1]
                    if not all(np.array_equal(labels, reference[s][1]) for s in seeds):
                        raise SystemExit(f"{dataset}: {name} and {reference_name} do not share test nodes")
                    ci = paired_bootstrap(labels, [arm[s][2] for s in seeds], [reference[s][2] for s in seeds],
                                          [arm[s][0]["test"]["threshold"] for s in seeds],
                                          [reference[s][0]["test"]["threshold"] for s in seeds])
                for metric in metrics:
                    result = compare(arm, reference, metric)
                    test = {"arm": name, "reference": reference_name, "metric": metric, **result, "ci95": ci.get(metric)}
                    tests.append(test)
                    if result["p"] is not None:
                        p_values[str(len(tests) - 1)] = result["p"]
        adjusted = holm(p_values)
        lines += ["", "| Arm | Against | Metric | Δ (seeds better) | 95% CI | p | Holm |", "|---|---|---|---|---|---|---|"]
        for index, test in enumerate(tests):
            interval = f"[{test['ci95'][0]:+.2f}, {test['ci95'][1]:+.2f}]" if test["ci95"] else "n/a"
            p = f"{test['p']:.4f}" if test["p"] is not None else "n/a"
            corrected = adjusted.get(str(index))
            test["holm"] = corrected
            shown = f"{corrected:.4f}" if corrected is not None else "n/a"
            delta = f"{test['delta']:+.2f} ({test['better']}/{len(test['seeds'])})" if test["delta"] is not None else "n/a"
            lines.append(f"| {test['arm']} | {test['reference']} | {LABEL[test['metric']]} | {delta} | {interval} | {p} | {shown} |")
        lines.append("")
        output[dataset] = tests
    (ROOT / "docs" / "DGP_PLUS_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "docs" / "dgp_plus_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
