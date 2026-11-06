"""First-token fine-tuning with several epoch-selection rules on one trajectory.

AdamW, batch size 4, 10 epochs. Every epoch runs and each rule keeps its own epoch: DGP+ picks by validation AUROC and
still reports the other rules. train() receives no test data, so neither the epoch nor anything else can depend on the
test set.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from dgp_plus.classifier import LabelTokens, first_token_loss, fraud_probability
from dgp_plus.evaluate import compute_metrics


@dataclass
class TrainResult:
    best_epoch: int = -1
    best_val_loss: float = math.inf
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    rule_epochs: dict[str, int] = field(default_factory=dict)    # multi-rule mode: rule -> kept epoch
    epoch_states: dict[int, dict] = field(default_factory=dict)  # model state of each kept epoch
    epoch_val_probabilities: dict[int, np.ndarray] = field(default_factory=dict)


def parse_rule(rule: str) -> tuple[str, int | None]:
    """'loss_es2' -> ('val_loss', 2): early stopping with patience 2. 'auroc_all' -> ('val_auroc', None): no stopping."""
    metric, _, stop = rule.partition("_")
    if metric not in ("loss", "auroc", "auprc") or not (stop == "all" or (stop.startswith("es") and stop[2:].isdigit())):
        raise ValueError(f"unknown selection rule {rule!r}; expected e.g. loss_es2, auroc_es3, auroc_all")
    return f"val_{metric}", None if stop == "all" else int(stop[2:])


class SelectionRule:
    """Follows one rule along a training run."""

    def __init__(self, name: str):
        self.name = name
        self.key, self.patience = parse_rule(name)
        self.lower_is_better = self.key == "val_loss"
        self.best_epoch, self.best = -1, math.inf if self.lower_is_better else -math.inf
        self.waited, self.stopped = 0, False

    def update(self, epoch: int, record: dict) -> None:
        if self.stopped:
            return
        value = record[self.key]
        if (value < self.best) if self.lower_is_better else (value > self.best):
            self.best_epoch, self.best, self.waited = epoch, value, 0
        else:
            self.waited += 1
            if self.patience is not None and self.waited >= self.patience:
                self.stopped = True


@torch.no_grad()
def predict(model, prompts: list[str], tokens: LabelTokens, batch_size: int) -> np.ndarray:
    """Fraud probability for each prompt (Eq. 14)."""
    model.set_training(False)
    probabilities = [fraud_probability(model.last_token_logits(prompts[i:i + batch_size]), tokens).cpu()
                     for i in range(0, len(prompts), batch_size)]
    return torch.cat(probabilities).numpy()


@torch.no_grad()
def validate(model, prompts: list[str], labels: np.ndarray, tokens: LabelTokens, batch_size: int) -> tuple[float, np.ndarray]:
    """Mean loss and probabilities from a single pass over the prompts."""
    model.set_training(False)
    total, probabilities = 0.0, []
    for start in range(0, len(prompts), batch_size):
        logits = model.last_token_logits(prompts[start:start + batch_size])
        batch_labels = torch.as_tensor(labels[start:start + batch_size], device=logits.device)
        total += float(first_token_loss(logits, batch_labels, tokens)) * len(batch_labels)
        probabilities.append(fraud_probability(logits, tokens).cpu())
    return total / max(len(prompts), 1), torch.cat(probabilities).numpy()


def train(model, train_prompts: list[str], train_labels: np.ndarray, val_prompts: list[str], val_labels: np.ndarray,
          tokens: LabelTokens, cfg: dict, seed: int, log_path: str | Path | None = None) -> TrainResult:
    """Train, then leave the model at the primary rule's epoch."""
    rng = np.random.default_rng(seed)
    parameters = model.parameters_to_train()
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    train_labels, val_labels = np.asarray(train_labels), np.asarray(val_labels)
    batch_size = cfg["batch_size"]
    result = TrainResult()
    rules = [SelectionRule(name) for name in cfg["selection_rules"]]
    started = time.time()

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.set_training(True)
        order = rng.permutation(len(train_prompts))
        running, steps = 0.0, 0
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            logits = model.last_token_logits([train_prompts[i] for i in chosen])
            loss = first_token_loss(logits, torch.as_tensor(train_labels[chosen], device=logits.device), tokens)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg["max_grad_norm"])
            optimizer.step()
            running += float(loss.detach())
            steps += 1

        val_loss, val_probabilities = validate(model, val_prompts, val_labels, tokens, cfg["eval_batch_size"])
        record = {"epoch": epoch, "train_loss": running / max(steps, 1), "val_loss": val_loss,
                  **{f"val_{k}": v for k, v in compute_metrics(val_labels, val_probabilities).items()}}
        result.history.append(record)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        for rule in rules:
            rule.update(epoch, record)
        kept = {rule.best_epoch for rule in rules}
        if epoch in kept:
            result.epoch_states[epoch] = model.state()
            result.epoch_val_probabilities[epoch] = val_probabilities
        for old in [e for e in result.epoch_states if e not in kept]:
            del result.epoch_states[old], result.epoch_val_probabilities[old]

    result.rule_epochs = {rule.name: rule.best_epoch for rule in rules}
    result.best_epoch = result.rule_epochs[cfg["primary_rule"]]
    result.best_val_loss = result.history[result.best_epoch - 1]["val_loss"]
    model.load_state(result.epoch_states[result.best_epoch])
    result.seconds = time.time() - started
    return result
