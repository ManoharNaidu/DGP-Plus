# DGP+

DGP+ detects fraud by combining a graph with a language model. It builds on DGP, the dual-granularity prompting method
from "DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs" (AAAI 2026). This is an
independent implementation, not the DGP authors' code.

DGP+ makes three changes to DGP:

| Change | DGP | DGP+ |
|---|---|---|
| Metapath hops | K = 2 | K = 1, with K = 2 kept as a comparison |
| Summary writer and classifier | Qwen3-8B | Gemma-2-9B |
| Epoch selection | Lowest validation loss | Best validation AUROC, patience 3 |

Metapath trimming with the Markov diffusion kernel, top-M = 4, the numeric summary, the prompt, LoRA, the first-token
Yes/No readout and the validation threshold are unchanged.

## Datasets

DGP+ works on five datasets:

| Dataset | Nodes | Task | Built by `prepare_data.py` |
|---|---:|---|---|
| AmazonVideo | 37,126 | Fraud reviews | Yes |
| YelpChi | 67,395 | Fraud reviews | Yes |
| Instagram | 7,946 | Commercial users | No |
| Reddit | 18,389 | Popular users | No |
| Amazon Instruments | 11,944 | Fraud users | No |

No data ships with this repository, because each dataset keeps its own terms.

- **AmazonVideo:** the source URL and checksum are in `configs/datasets/amazonvideo.yaml`.
- **YelpChi:** this needs the original release with raw text, as `reviews.jsonl`, not `YelpChi.mat`.
- **Instagram, Reddit and Amazon Instruments:** these are not built here. Place each one in `data/processed/<name>/`:

```text
graph.json
labels.npy
numeric.npy
relation_<name>.npz    one file per relation
split.json
texts.jsonl
```

## Install

Python 3.12 and a GPU are required. Gemma-2-9B is gated on Hugging Face, so accept its terms and log in first.

```bash
pip install -e ".[dev]"
hf auth login
pytest
```

## Run

Each dataset has a K = 1 and a K = 2 config in `configs/runs/`. Shared settings are in `configs/base.yaml`.

```bash
python scripts/prepare_data.py --config configs/runs/amazonvideo_k1.yaml
python scripts/run_queue.py configs/runs/*.yaml --gpus 0 1 2 3
python scripts/evaluate.py configs/runs/*.yaml
```

Every run writes its metrics, per-node predictions and a manifest to `runs/results/raw/`. The epoch and threshold are
chosen on validation data only.

To compare with another set of runs on the same test nodes:

```bash
python scripts/compare_arms.py --baseline baseline/
```

## Future work

- A standalone builder for Instagram, Reddit and Amazon Instruments, so that all five datasets can be built from this
  repository.

## Licence

The code is MIT licensed; see `LICENSE`. Gemma-2-9B, DeBERTaV3 and the datasets keep their own terms.
