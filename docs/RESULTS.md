# DGP reproduction results

These are the reference runs that DGP+ is compared with. They follow the paper recipe: Qwen3-8B writer and classifier,
K = 2, lowest validation loss with patience 2. They were produced by a separate reproduction code base, which is not
part of this repository, so this repository cannot rerun them. Their `metrics.json` and `predictions.csv` files are in the
`baseline/` folder of the Hugging Face mirror.

Test set. Macro-F1 / AUROC / AUPRC, mean ± population std over seeds 0-4. Epoch and threshold chosen on validation only.

| Dataset | Reproduced | Reference | Gap (reproduced - reference) |
|---|---|---|---|
| AmazonVideo | 65.16 ± 0.52 / 75.07 ± 0.50 / 32.33 ± 0.65 | Paper 66.91 / 77.32 / 34.63 | -1.75 / -2.25 / -2.30 |
| YelpChi | 61.76 ± 1.86 / 78.38 ± 0.88 / 31.05 ± 1.31 | Paper 69.07 / 84.28 / 48.87 | -7.31 / -5.90 / -17.82 |
| Instagram | 52.75 ± 2.10 / 58.27 ± 2.64 / 12.46 ± 1.08 | FLAG 53.41 / 59.01 / n/a | -0.66 / -0.74 / n/a |
| Reddit | 53.61 ± 1.01 / 62.26 ± 1.52 / 12.92 ± 0.81 | FLAG 53.66 / 62.25 / n/a | -0.05 / +0.01 / n/a |
| Amazon Instruments | 86.25 ± 1.01 / 97.02 ± 0.32 / 80.27 ± 1.74 | FLAG 76.45 / 89.95 / n/a | +9.80 / +7.07 / n/a |

The reproduction does not reach the paper on its two public datasets. FLAG figures are means across 7 GNN backbones, so
they are context only, not a head-to-head result.

## Per seed

| Dataset | Metric | Seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| AmazonVideo | Macro-F1 | 65.70 | 64.90 | 64.26 | 65.46 | 65.49 |
| | AUROC | 75.13 | 75.54 | 74.72 | 74.33 | 75.65 |
| | AUPRC | 32.78 | 32.47 | 31.70 | 31.49 | 33.20 |
| YelpChi | Macro-F1 | 61.97 | 58.45 | 62.72 | 61.58 | 64.09 |
| | AUROC | 78.80 | 76.67 | 78.40 | 79.08 | 78.94 |
| | AUPRC | 31.60 | 28.48 | 31.30 | 32.03 | 31.83 |
| Instagram | Macro-F1 | 53.63 | 50.31 | 54.56 | 55.07 | 50.18 |
| | AUROC | 56.85 | 53.79 | 60.52 | 60.81 | 59.40 |
| | AUPRC | 11.83 | 10.65 | 13.35 | 13.57 | 12.89 |
| Reddit | Macro-F1 | 54.74 | 53.06 | 51.94 | 54.41 | 53.89 |
| | AUROC | 62.57 | 60.97 | 61.06 | 65.08 | 61.62 |
| | AUPRC | 13.14 | 12.34 | 12.09 | 14.38 | 12.65 |
| Amazon Instruments | Macro-F1 | 87.94 | 86.71 | 84.99 | 86.02 | 85.59 |
| | AUROC | 97.40 | 97.03 | 96.54 | 97.33 | 96.79 |
| | AUPRC | 82.45 | 81.11 | 77.90 | 81.34 | 78.58 |

## Amazon Instruments check

The score is far above the FLAG figure, so it was checked for leakage. No leak was found, but the check is not exhaustive.

- The prompt templates carry no label field.
- The source text has no meaningful label words: "fraud" and "spam" are at or near 0% in both classes.
- The text itself is a strong signal. TF-IDF with logistic regression, trained on the train split, reaches 92.72 AUROC and
  57.47 AUPRC on test.
- Word count alone gives 83.43 AUROC. Labels in this dataset come from CARE-GNN, so text may be a proxy for how the labels
  were made rather than for fraud. Not verified.
- 2,437 of 8,639 target texts (28.2%) are cut to 1,024 tokens in the final prompt, on top of the 3,900-token cap applied
  before summarising.

## Code parity

The implementation was checked against the reproduction code before any DGP+ run.

- Prompts: byte-identical on all 61,172 prompts across the five datasets.
- Model, paper recipe seed 0 on AmazonVideo: 65.72 / 75.10 / 32.69 against 64.78 / 75.13 / 32.78.
- The tolerance was 0.92 / 0.11 / 0.34. AUROC and AUPRC pass. Macro-F1 is 0.01 outside it, with a gap of 0.93.
- The two runs chose thresholds of 0.25 and 0.30, and best epoch 2 in both. Test prediction correlation with 7 stored
  same-config runs is 0.9930-0.9971, against 0.9901-0.9967 between those runs. The overshoot was treated as seed noise and
  not rerun. Only one seed was compared.
- Gemma classifier, one AmazonVideo seed: predictions byte-identical on 8,724 nodes.
- The parity tools needed the reproduction's stored outputs, so they are not part of this repository.

After the code was trimmed for release, Instagram K = 1 seed 0 was rerun. Macro-F1 / AUROC / AUPRC match the original run,
and all 7,152 predictions match within the 6-decimal precision of the original file.
