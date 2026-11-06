import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from dgp_plus import evaluate as ev
from dgp_plus.classifier import MockClassifier
from dgp_plus.config import load_config
from dgp_plus.pipeline import build_prompts
from dgp_plus.runs import run_seed, summarise_runs
from dgp_plus.train import SelectionRule, parse_rule, train


def test_threshold_comes_from_validation_and_uses_strict_greater_than():
    assert ev.macro_f1_at([0, 1], [0.5, 0.6], 0.5) == 100
    labels = np.array([0, 0, 1, 1])
    assert ev.select_threshold(labels, np.array([0.1, 0.2, 0.8, 0.9])) == ev.THRESHOLDS[3]


def test_aggregate_uses_population_std():
    out = ev.aggregate([{"macro_f1": 1.0, "auroc": 2.0, "auprc": 3.0}, {"macro_f1": 3.0, "auroc": 2.0, "auprc": 3.0}])
    assert out["macro_f1"] == {"mean": 2.0, "std": 1.0}


def test_holm():
    assert ev.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_bootstrap_covers_macro_f1_with_fixed_thresholds():
    labels = np.array([0, 1] * 20)
    scores = labels * 0.6 + 0.2
    ci = ev.paired_bootstrap(labels, [scores], [1 - scores], [0.5], [0.5])
    assert ci["macro_f1"][0] > 90
    assert "macro_f1" not in ev.paired_bootstrap(labels, [scores], [scores])


def test_bootstrap_of_identical_runs_is_zero():
    labels = np.array([0, 1] * 30)
    scores = np.random.default_rng(0).random(60)
    ci = ev.paired_bootstrap(labels, [scores], [scores])
    assert ci == {"auroc": [0.0, 0.0], "auprc": [0.0, 0.0]}


def test_full_path_writes_every_record(tiny_config):
    prompts = build_prompts(tiny_config, "cpu")
    metrics = run_seed(tiny_config, prompts, 0, "cpu")
    run_dir = next((Path(tiny_config["paths"]["results"]) / "raw").iterdir())
    names = {p.name for p in run_dir.iterdir()}
    assert {"metrics.json", "predictions.csv", "run_manifest.json", "training_log.jsonl"} <= names
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert {"config", "git_commit", "seed", "peak_memory", "runtime_seconds", "environment"} <= set(manifest)
    assert {"mean", "max"} <= set(manifest["prompt_tokens"])
    assert summarise_runs(tiny_config)["seeds"] == [0]
    assert metrics["test"]["threshold"] == metrics["val"]["threshold"]


def test_finished_run_is_skipped(tiny_config):
    prompts = build_prompts(tiny_config, "cpu")
    first = run_seed(tiny_config, prompts, 1, "cpu")
    assert run_seed(tiny_config, prompts, 1, "cpu") == first


def test_prompts_rebuild_identically_from_cache(tiny_config):
    assert build_prompts(tiny_config, "cpu").prompts == build_prompts(tiny_config, "cpu").prompts


def test_numeric_summary_switch_lives_in_config(tiny_config):
    tiny_config["dgp"]["numeric_summary"] = False
    assert not any("neighbor mean" in p for p in build_prompts(tiny_config, "cpu").prompts.values())


def test_no_test_set_leakage(tiny_config):
    """Flipping every test label and changing test prompts must not move the chosen epoch or threshold."""
    prompts = build_prompts(tiny_config, "cpu")
    clean = run_seed(tiny_config, prompts, 0, "cpu")

    other = {**tiny_config, "paths": {**tiny_config["paths"], "results": tiny_config["paths"]["results"] + "_altered"}}
    prompts.graph.labels[prompts.split.test] = 1 - prompts.graph.labels[prompts.split.test]
    for node in prompts.split.test:
        prompts.prompts[int(node)] = "altered text about something else entirely"
    altered = run_seed(other, prompts, 0, "cpu")

    assert altered["best_epoch"] == clean["best_epoch"]
    assert altered["val"]["threshold"] == clean["val"]["threshold"]
    assert altered["val"]["auroc"] == clean["val"]["auroc"]
    assert altered["test"]["auroc"] != clean["test"]["auroc"]


def test_rule_names_are_parsed_and_bad_ones_refused():
    assert parse_rule("loss_es2") == ("val_loss", 2)
    assert parse_rule("auroc_all") == ("val_auroc", None)
    with pytest.raises(ValueError):
        parse_rule("accuracy_es2")


@pytest.mark.parametrize("name,expected", [("loss_es2", 2), ("auroc_es3", 3), ("auroc_all", 7)])
def test_each_rule_keeps_the_expected_epoch(name, expected):
    losses = [0.5, 0.4, 0.45, 0.46, 0.3, 0.25, 0.2]
    aurocs = [60, 62, 63, 62, 62, 62, 64]
    rule = SelectionRule(name)
    for epoch, (loss, auroc) in enumerate(zip(losses, aurocs), 1):
        rule.update(epoch, {"val_loss": loss, "val_auroc": auroc})
    assert rule.best_epoch == expected


def test_multi_rule_run_scores_each_rule_without_touching_test_during_training(tmp_path):
    prompts = [f"word{i % 2} text {i}" for i in range(16)]
    labels = np.array([i % 2 for i in range(16)])
    model = MockClassifier()
    cfg = {"learning_rate": 0.05, "weight_decay": 0.0, "batch_size": 4, "max_epochs": 3, "max_grad_norm": 1.0,
           "eval_batch_size": 4, "selection_rules": ["auroc_es3", "loss_es2", "auroc_all"],
           "primary_rule": "auroc_es3"}
    result = train(model, prompts, labels, prompts, labels, model.labels, cfg, seed=0)
    assert len(result.history) == 3                                   # every epoch runs
    assert set(result.rule_epochs) == set(cfg["selection_rules"])
    assert result.best_epoch == result.rule_epochs["auroc_es3"]
    assert set(result.epoch_states) == set(result.rule_epochs.values())   # nothing else is kept
    assert "test" not in inspect.signature(train).parameters


def test_configs_hold_the_three_tweaks():
    root = Path(__file__).resolve().parents[1] / "configs" / "runs"
    one, two = load_config(root / "amazonvideo_k1.yaml"), load_config(root / "amazonvideo_k2.yaml")
    assert one["dgp"]["K"] == 1 and two["dgp"]["K"] == 2
    assert one["classifier"]["model_id"] == one["summarizer"]["model_id"] == "google/gemma-2-9b-it"
    assert one["training"]["primary_rule"] == "auroc_es3"
    assert {k: v for k, v in one.items() if k != "dgp"} == {k: v for k, v in two.items() if k != "dgp"}
