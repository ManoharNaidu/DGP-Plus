"""Fill the caches up to a stage. Shards let many GPUs share one store.

    python scripts/generate_summaries.py --config configs/datasets/reddit.yaml --until trim --device cuda:0
    python scripts/generate_summaries.py --config ... --until node_summaries --shard 0/8 --device cuda:0

Build the trims in one process before starting shards, and build prompts in one process after them.
"""

from _common import config_from, parser

from dgp_plus.pipeline import STAGES, build_prompts


def main():
    made = parser(__doc__)
    made.add_argument("--until", default="prompts", choices=STAGES)
    made.add_argument("--device", default="cuda:0")
    made.add_argument("--shard", help="i/n: generate only this slice of the keys")
    args = made.parse_args()
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    result = build_prompts(config_from(args), args.device, args.until, shard)
    if result:
        print(f"{len(result.prompts)} prompts, mean {result.mean_tokens:.1f} tokens, max {result.max_tokens}")


if __name__ == "__main__":
    main()
