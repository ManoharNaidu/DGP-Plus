"""Train and evaluate seeds. Finished seeds are skipped, so this command can be repeated safely.

    python scripts/train.py --config configs/datasets/instagram.yaml --seeds 0 --device cuda:0
"""

import json

from _common import config_from, parser

from dgp_plus.pipeline import build_prompts
from dgp_plus.runs import run_seed, summarise_runs


def main():
    made = parser(__doc__)
    made.add_argument("--seeds", type=int, nargs="+", help="default: the config's seeds")
    made.add_argument("--device", default="cuda:0")
    args = made.parse_args()
    cfg = config_from(args)
    prompts = build_prompts(cfg, args.device)
    for seed in args.seeds if args.seeds is not None else cfg["seeds"]:
        print(f"seed {seed}: {json.dumps(run_seed(cfg, prompts, seed, args.device)['test'])}", flush=True)
    print(json.dumps(summarise_runs(cfg), indent=2))


if __name__ == "__main__":
    main()
