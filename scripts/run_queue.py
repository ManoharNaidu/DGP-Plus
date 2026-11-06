"""Run every stage for several datasets across GPUs, one job per GPU. Resumable.

    python scripts/run_queue.py configs/runs/reddit_k1.yaml configs/runs/instagram_k1.yaml --gpus 0 1 2 3
    python scripts/run_queue.py configs/runs/*.yaml --only summaries    # stop before training

Stages for each dataset: trims, node summary shards, metapath summary shards, then one training job per seed. A free
GPU takes the first ready job. A failed job retries twice; after that its dependants never start and other datasets
carry on. Configs of one dataset that share a writer share a node summary store, so they run their stages in the order
given rather than generating the same text twice. (Named run_queue.py because a script called queue.py would hide Python's own queue module.)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgp_plus.config import load_config  # noqa: E402

MAX_TRIES = 3


def jobs_for(config: str, shards: int) -> list[dict]:
    cfg = load_config(config)
    name = Path(config).stem

    def generate(job: str, until: str, deps: list[str], shard: str | None = None) -> dict:
        extra = ["--shard", shard] if shard else []
        return {"name": job, "deps": deps, "tries": 0, "command": ["scripts/generate_summaries.py", "--config", config,
                                                                   "--until", until, *extra]}

    nodes = [f"{name}_node_{i}" for i in range(shards)]
    metas = [f"{name}_meta_{i}" for i in range(shards)]
    jobs = [generate(f"{name}_trim", "trim", [])]
    jobs += [generate(nodes[i], "node_summaries", [f"{name}_trim"], f"{i}/{shards}") for i in range(shards)]
    jobs += [generate(metas[i], "metapath_summaries", nodes, f"{i}/{shards}") for i in range(shards)]
    jobs.append(generate(f"{name}_prompts", "prompts", metas))
    jobs += [{"name": f"{name}_train_s{seed}", "deps": [f"{name}_prompts"], "tries": 0,
              "command": ["scripts/train.py", "--config", config, "--seeds", str(seed)]} for seed in cfg["seeds"]]
    return jobs


def main():
    made = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    made.add_argument("configs", nargs="+")
    made.add_argument("--gpus", type=int, nargs="+", default=[0])
    made.add_argument("--shards", type=int, default=8)
    made.add_argument("--out", default="runs/queue")
    made.add_argument("--only", default="all", choices=["all", "summaries"], help="summaries: leave out training jobs")
    args = made.parse_args()
    out = Path(args.out)
    (out / "done").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    def event(**fields):
        with open(out / "status.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}) + "\n")

    def done(name: str) -> bool:
        return (out / "done" / name).exists()

    jobs, last_nodes = [], {}
    for config in args.configs:
        made_jobs = jobs_for(config, args.shards)
        cfg = load_config(config)
        store = (cfg["dataset"]["name"], cfg["summarizer"]["model_id"])
        made_jobs[0]["deps"] += last_nodes.get(store, [])     # the trim job waits for the earlier arm's node shards
        last_nodes[store] = [j["name"] for j in made_jobs if "_node_" in j["name"]]
        jobs += [j for j in made_jobs if args.only == "all" or "_train_" not in j["name"]]
    running, failed = {}, set()
    event(event="queue_start", jobs=len(jobs))
    while True:
        for gpu, (job, process, started, log) in list(running.items()):
            if process.poll() is None:
                continue
            log.close()
            del running[gpu]
            event(event="end", job=job["name"], gpu=gpu, rc=process.returncode, seconds=round(time.time() - started))
            if process.returncode == 0:
                (out / "done" / job["name"]).touch()
            elif job["tries"] >= MAX_TRIES:
                failed.add(job["name"])
                event(event="gave_up", job=job["name"])
        active = {entry[0]["name"] for entry in running.values()}
        ready = [j for j in jobs if not done(j["name"]) and j["name"] not in active | failed
                 and all(done(d) for d in j["deps"])]
        ready.sort(key=lambda j: "_train_" in j["name"])     # summaries first, so no arm waits behind hours of training
        for gpu in [g for g in args.gpus if g not in running]:
            if not ready:
                break
            job = ready.pop(0)
            job["tries"] += 1
            log = open(out / "logs" / f"{job['name']}.log", "a", encoding="utf-8")
            process = subprocess.Popen([sys.executable, "-W", "ignore", *job["command"], "--device", "cuda:0"], cwd=ROOT,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                                            "TOKENIZERS_PARALLELISM": "false"},
                                       stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (job, process, time.time(), log)
            event(event="start", job=job["name"], gpu=gpu, attempt=job["tries"])
        if not running:
            event(event="queue_end", unfinished=sum(not done(j["name"]) for j in jobs), failed=sorted(failed))
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
