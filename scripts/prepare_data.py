"""Build a processed graph from raw files, check it against the expected counts, and create its split once.

    python scripts/prepare_data.py --config configs/runs/amazonvideo_k1.yaml

Only AmazonVideo and YelpChi are built here. Reddit, Instagram and Amazon Instruments must already be in
data/processed/<name>/ with their split.json; see "Data" in the README.
"""

from pathlib import Path

from _common import config_from, parser

from dgp_plus.data import load_graph, make_split
from dgp_plus.raw_data import build_amazonvideo, build_yelpchi


def main():
    cfg = config_from(parser(__doc__).parse_args())
    dataset = cfg["dataset"]
    directory = Path(cfg["paths"]["data"]) / dataset["name"]
    if (directory / "graph.json").exists():
        graph = load_graph(directory)
    elif dataset["name"] == "amazonvideo":
        graph = build_amazonvideo(dataset["raw"]["path"], {**dataset["raw"], "relations": dataset["relations"]})
    elif dataset["name"] == "yelpchi":
        graph = build_yelpchi(dataset["raw"]["path"], {"relations": dataset["relations"]})
    else:
        raise SystemExit(f"{directory} has no graph; see this script's docstring")
    summary = graph.summary()
    wrong = {k: (v, summary[k]) for k, v in dataset["expected"].items() if summary[k] != v}
    if wrong:
        raise SystemExit(f"counts differ from expected (expected, built): {wrong}")
    graph.save(directory)
    if not (directory / "split.json").exists():
        make_split(graph.labels, dataset["split"], dataset["split"]["seed"]).save(directory / "split.json")
    print(summary)


if __name__ == "__main__":
    main()
