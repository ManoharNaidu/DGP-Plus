"""One run: train a seed, evaluate it, and record everything needed to trace the result.

A run's folder holds metrics.json, predictions.csv, training_log.jsonl and run_manifest.json. A finished run is
skipped, so any command can be repeated safely. With several selection rules, the test set is scored once per distinct
kept epoch, after training has ended, and metrics.json gains a "selection" entry per rule.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dgp_plus.classifier import make_classifier
from dgp_plus.config import config_hash
from dgp_plus.evaluate import aggregate, evaluate_split
from dgp_plus.pipeline import PromptSet
from dgp_plus.train import predict, train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    """Current commit, marked -dirty with uncommitted changes, so a commit alone is never over-trusted."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        if commit.returncode != 0:
            return "not_a_git_repository"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10)
        return commit.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "git_unavailable"


def environment() -> dict:
    import peft
    import scipy
    import sklearn
    import transformers

    info = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return info


def peak_memory() -> dict:
    out = {"cpu_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)}
    if torch.cuda.is_available():
        out["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    return out


def run_seed(cfg: dict, prompts: PromptSet, seed: int, device: str) -> dict:
    """Train one seed and evaluate it. The epoch and threshold come from validation data only."""
    name = cfg["dataset"]["name"]
    run_id = f"{name}_seed{seed}_{config_hash(cfg)}"
    run_dir = Path(cfg["paths"]["results"]) / "raw" / run_id
    if (run_dir / "metrics.json").exists():
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_log.jsonl").unlink(missing_ok=True)     # a stopped run restarts from epoch 1
    started = time.time()
    set_seed(seed)
    model = make_classifier(cfg["classifier"], device)
    graph, split = prompts.graph, prompts.split
    nodes = {"train": split.train, "val": split.val, "test": split.test}
    text = {k: [prompts.prompts[int(v)] for v in ids] for k, ids in nodes.items()}
    labels = {k: graph.labels[ids].astype(int) for k, ids in nodes.items()}

    training = cfg["training"]
    result = train(model, text["train"], labels["train"], text["val"], labels["val"], model.labels, training, seed,
                   log_path=run_dir / "training_log.jsonl")

    def write_predictions(path: Path, scores: dict) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node", "split", "label", "p_fraud"])
            for split_name in ("val", "test"):
                for node, label, p in zip(nodes[split_name], labels[split_name], scores[split_name]):
                    writer.writerow([int(node), split_name, int(label), f"{p:.9g}"])

    selection = {}
    for epoch in sorted(set(result.rule_epochs.values()) - {result.best_epoch}):   # the primary epoch is scored last
        model.load_state(result.epoch_states[epoch])
        scores = {"val": result.epoch_val_probabilities[epoch],
                  "test": predict(model, text["test"], model.labels, training["eval_batch_size"])}
        write_predictions(run_dir / f"predictions_epoch{epoch}.csv", scores)
        selection[epoch] = evaluate_split(labels["val"], scores["val"], labels["test"], scores["test"])
    model.load_state(result.epoch_states[result.best_epoch])
    probabilities = {k: predict(model, text[k], model.labels, training["eval_batch_size"]) for k in ("val", "test")}
    metrics = {"seed": seed, "best_epoch": result.best_epoch,
               **evaluate_split(labels["val"], probabilities["val"], labels["test"], probabilities["test"])}
    write_predictions(run_dir / "predictions.csv", probabilities)
    selection[result.best_epoch] = {"val": metrics["val"], "test": metrics["test"]}
    metrics["selection"] = {rule: {"epoch": epoch, **selection[epoch]} for rule, epoch in result.rule_epochs.items()}
    manifest = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": run_id, "seed": seed, "device": device,
                "git_commit": git_commit(), "environment": environment(), "config": cfg, "best_epoch": result.best_epoch,
                "runtime_seconds": round(time.time() - started, 1), "train_seconds": round(result.seconds, 1),
                "peak_memory": peak_memory(), "split_sizes": split.sizes(), "dataset": graph.summary(),
                "prompt_tokens": {"mean": prompts.mean_tokens, "max": prompts.max_tokens,
                                  "truncated_targets": prompts.truncated_targets}}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")   # written last: marks done
    if training.get("save_adapter", True):
        model.save(Path(cfg["paths"]["results"]) / "checkpoints" / run_id)
    return metrics


def finished_runs(cfg: dict) -> list[dict]:
    """metrics.json of every finished seed of this exact config."""
    pattern = f"{cfg['dataset']['name']}_seed*_{config_hash(cfg)}/metrics.json"
    paths = sorted((Path(cfg["paths"]["results"]) / "raw").glob(pattern))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def summarise_runs(cfg: dict) -> dict | None:
    runs = finished_runs(cfg)
    return {"dataset": cfg["dataset"]["name"], "seeds": [r["seed"] for r in runs],
            "test": aggregate([r["test"] for r in runs])} if runs else None
