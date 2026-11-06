# DGP+ results

Test set, mean ± population std over seeds 0-4. Epoch and threshold chosen on validation only.
Δ = arm − reference, mean over paired seeds. CI = paired bootstrap 95% over test nodes, 2,000 rounds.
p = paired t over seeds. Holm is applied over all tests of one dataset.
A comparison with fewer than 5 paired seeds is shown as n/a.

## amazonvideo

| Arm | Seeds | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |
|---|---|---|---|---|---|
| DGP reproduced | 5 | 65.16 ± 0.52 | 75.07 ± 0.50 | 32.33 ± 0.65 | |
| DGP+ K=1 | 5 | 65.05 ± 1.45 | 76.56 ± 0.25 | 34.01 ± 0.77 | 223.0 |
| DGP+ K=2 | 5 | 64.86 ± 0.83 | 76.27 ± 0.37 | 34.31 ± 1.00 | 540.0 |
| DGP+ K=2 loss_es2 | 5 | 65.04 ± 1.01 | 76.38 ± 0.34 | 34.31 ± 1.00 | 540.0 |
| DGP+ K=2 auroc_all | 5 | 64.86 ± 0.83 | 76.27 ± 0.37 | 34.31 ± 1.00 | 540.0 |

| Arm | Against | Metric | Δ (seeds better) | 95% CI | p | Holm |
|---|---|---|---|---|---|---|
| DGP+ K=1 | DGP reproduced | Macro-F1 | -0.12 (3/5) | [-0.74, +0.46] | 0.8639 | 1.0000 |
| DGP+ K=1 | DGP reproduced | AUROC | +1.49 (5/5) | [+0.75, +2.27] | 0.0010 | 0.0078 |
| DGP+ K=1 | DGP reproduced | AUPRC | +1.68 (5/5) | [+0.45, +2.92] | 0.0040 | 0.0279 |
| DGP+ K=1 | DGP+ K=2 | Macro-F1 | +0.19 (3/5) | [-0.25, +0.67] | 0.8342 | 1.0000 |
| DGP+ K=1 | DGP+ K=2 | AUROC | +0.29 (5/5) | [-0.09, +0.69] | 0.0358 | 0.1790 |
| DGP+ K=1 | DGP+ K=2 | AUPRC | -0.30 (2/5) | [-0.94, +0.35] | 0.5123 | 1.0000 |
| DGP+ K=2 | DGP reproduced | Macro-F1 | -0.30 (3/5) | [-0.95, +0.29] | 0.6405 | 1.0000 |
| DGP+ K=2 | DGP reproduced | AUROC | +1.20 (5/5) | [+0.48, +1.90] | 0.0006 | 0.0058 |
| DGP+ K=2 | DGP reproduced | AUPRC | +1.98 (5/5) | [+0.72, +3.22] | 0.0049 | 0.0292 |

## yelpchi

| Arm | Seeds | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |
|---|---|---|---|---|---|
| DGP reproduced | 5 | 61.76 ± 1.86 | 78.38 ± 0.88 | 31.05 ± 1.31 | |
| DGP+ K=1 | 5 | 71.40 ± 0.64 | 85.30 ± 0.63 | 56.57 ± 1.49 | 268.1 |
| DGP+ K=2 | 5 | 71.44 ± 0.19 | 85.55 ± 0.35 | 55.98 ± 0.99 | 543.2 |
| DGP+ K=2 loss_es2 | 5 | 68.00 ± 4.32 | 82.34 ± 4.00 | 45.15 ± 13.04 | 543.2 |
| DGP+ K=2 auroc_all | 5 | 71.44 ± 0.19 | 85.55 ± 0.35 | 55.98 ± 0.99 | 543.2 |

| Arm | Against | Metric | Δ (seeds better) | 95% CI | p | Holm |
|---|---|---|---|---|---|---|
| DGP+ K=1 | DGP reproduced | Macro-F1 | +9.63 (5/5) | [+8.45, +10.78] | 0.0005 | 0.0025 |
| DGP+ K=1 | DGP reproduced | AUROC | +6.92 (5/5) | [+6.10, +7.81] | 0.0003 | 0.0018 |
| DGP+ K=1 | DGP reproduced | AUPRC | +25.53 (5/5) | [+23.27, +27.74] | 0.0000 | 0.0001 |
| DGP+ K=1 | DGP+ K=2 | Macro-F1 | -0.04 (2/5) | [-0.40, +0.34] | 0.9151 | 0.9955 |
| DGP+ K=1 | DGP+ K=2 | AUROC | -0.25 (2/5) | [-0.51, +0.03] | 0.4977 | 0.9955 |
| DGP+ K=1 | DGP+ K=2 | AUPRC | +0.59 (4/5) | [+0.01, +1.16] | 0.1808 | 0.5424 |
| DGP+ K=2 | DGP reproduced | Macro-F1 | +9.67 (5/5) | [+8.47, +10.87] | 0.0005 | 0.0025 |
| DGP+ K=2 | DGP reproduced | AUROC | +7.17 (5/5) | [+6.35, +8.04] | 0.0000 | 0.0003 |
| DGP+ K=2 | DGP reproduced | AUPRC | +24.93 (5/5) | [+22.68, +27.15] | 0.0000 | 0.0000 |

## instagram

| Arm | Seeds | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |
|---|---|---|---|---|---|
| DGP reproduced | 5 | 52.75 ± 2.10 | 58.27 ± 2.64 | 12.46 ± 1.08 | |
| DGP+ K=1 | 5 | 53.86 ± 0.87 | 58.86 ± 1.19 | 14.18 ± 1.26 | 70.5 |
| DGP+ K=2 | 5 | 52.66 ± 2.59 | 60.20 ± 1.76 | 14.90 ± 1.48 | 90.3 |
| DGP+ K=2 loss_es2 | 5 | 51.71 ± 2.09 | 58.83 ± 1.26 | 13.58 ± 0.96 | 90.3 |
| DGP+ K=2 auroc_all | 5 | 52.20 ± 2.64 | 60.58 ± 1.97 | 14.86 ± 1.49 | 90.3 |

| Arm | Against | Metric | Δ (seeds better) | 95% CI | p | Holm |
|---|---|---|---|---|---|---|
| DGP+ K=1 | DGP reproduced | Macro-F1 | +1.11 (3/5) | [+0.14, +2.06] | 0.4863 | 1.0000 |
| DGP+ K=1 | DGP reproduced | AUROC | +0.58 (3/5) | [-1.08, +2.25] | 0.7140 | 1.0000 |
| DGP+ K=1 | DGP reproduced | AUPRC | +1.72 (4/5) | [+0.60, +2.95] | 0.1628 | 1.0000 |
| DGP+ K=1 | DGP+ K=2 | Macro-F1 | +1.19 (3/5) | [+0.52, +1.85] | 0.4968 | 1.0000 |
| DGP+ K=1 | DGP+ K=2 | AUROC | -1.34 (2/5) | [-2.38, -0.31] | 0.3772 | 1.0000 |
| DGP+ K=1 | DGP+ K=2 | AUPRC | -0.72 (3/5) | [-1.31, -0.10] | 0.6175 | 1.0000 |
| DGP+ K=2 | DGP reproduced | Macro-F1 | -0.09 (2/5) | [-1.07, +0.87] | 0.9315 | 1.0000 |
| DGP+ K=2 | DGP reproduced | AUROC | +1.93 (4/5) | [+0.25, +3.58] | 0.1747 | 1.0000 |
| DGP+ K=2 | DGP reproduced | AUPRC | +2.44 (5/5) | [+1.29, +3.69] | 0.0087 | 0.0781 |

## reddit

| Arm | Seeds | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |
|---|---|---|---|---|---|
| DGP reproduced | 5 | 53.61 ± 1.01 | 62.26 ± 1.52 | 12.92 ± 0.81 | |
| DGP+ K=1 | 5 | 53.58 ± 3.20 | 65.37 ± 1.89 | 15.17 ± 1.23 | 213.8 |
| DGP+ K=2 | 5 | 53.97 ± 1.08 | 65.01 ± 1.00 | 14.68 ± 0.68 | 233.4 |
| DGP+ K=2 loss_es2 | 5 | 53.73 ± 1.47 | 64.19 ± 1.36 | 13.81 ± 0.82 | 233.4 |
| DGP+ K=2 auroc_all | 5 | 53.97 ± 1.08 | 65.01 ± 1.00 | 14.68 ± 0.68 | 233.4 |

| Arm | Against | Metric | Δ (seeds better) | 95% CI | p | Holm |
|---|---|---|---|---|---|---|
| DGP+ K=1 | DGP reproduced | Macro-F1 | -0.03 (3/5) | [-0.86, +0.78] | 0.9886 | 1.0000 |
| DGP+ K=1 | DGP reproduced | AUROC | +3.11 (4/5) | [+1.78, +4.37] | 0.0575 | 0.3451 |
| DGP+ K=1 | DGP reproduced | AUPRC | +2.25 (4/5) | [+1.43, +3.07] | 0.0419 | 0.3349 |
| DGP+ K=1 | DGP+ K=2 | Macro-F1 | -0.39 (2/5) | [-1.00, +0.22] | 0.8387 | 1.0000 |
| DGP+ K=1 | DGP+ K=2 | AUROC | +0.36 (3/5) | [-0.38, +1.09] | 0.6966 | 1.0000 |
| DGP+ K=1 | DGP+ K=2 | AUPRC | +0.49 (3/5) | [-0.14, +1.11] | 0.3090 | 1.0000 |
| DGP+ K=2 | DGP reproduced | Macro-F1 | +0.36 (1/5) | [-0.41, +1.09] | 0.6974 | 1.0000 |
| DGP+ K=2 | DGP reproduced | AUROC | +2.75 (5/5) | [+1.51, +3.89] | 0.0241 | 0.2173 |
| DGP+ K=2 | DGP reproduced | AUPRC | +1.76 (5/5) | [+1.04, +2.48] | 0.0425 | 0.3349 |

## amazon_musical

| Arm | Seeds | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |
|---|---|---|---|---|---|
| DGP reproduced | 5 | 86.25 ± 1.01 | 97.02 ± 0.32 | 80.27 ± 1.74 | |
| DGP+ K=1 | 5 | 88.85 ± 1.01 | 97.67 ± 0.19 | 86.65 ± 1.09 | 628.6 |
| DGP+ K=2 | 5 | 88.54 ± 1.30 | 97.74 ± 0.14 | 85.68 ± 1.42 | 838.9 |
| DGP+ K=2 loss_es2 | 5 | 88.34 ± 1.29 | 97.70 ± 0.14 | 85.41 ± 1.65 | 838.9 |
| DGP+ K=2 auroc_all | 5 | 88.40 ± 1.26 | 97.69 ± 0.15 | 85.38 ± 1.49 | 838.9 |

| Arm | Against | Metric | Δ (seeds better) | 95% CI | p | Holm |
|---|---|---|---|---|---|---|
| DGP+ K=1 | DGP reproduced | Macro-F1 | +2.60 (5/5) | [+1.52, +3.67] | 0.0045 | 0.0351 |
| DGP+ K=1 | DGP reproduced | AUROC | +0.65 (5/5) | [+0.33, +0.96] | 0.0211 | 0.0925 |
| DGP+ K=1 | DGP reproduced | AUPRC | +6.37 (5/5) | [+4.32, +8.47] | 0.0024 | 0.0220 |
| DGP+ K=1 | DGP+ K=2 | Macro-F1 | +0.31 (2/5) | [-0.33, +0.92] | 0.7327 | 1.0000 |
| DGP+ K=1 | DGP+ K=2 | AUROC | -0.07 (3/5) | [-0.24, +0.08] | 0.5051 | 1.0000 |
| DGP+ K=1 | DGP+ K=2 | AUPRC | +0.97 (3/5) | [-0.03, +1.99] | 0.3511 | 1.0000 |
| DGP+ K=2 | DGP reproduced | Macro-F1 | +2.29 (5/5) | [+1.38, +3.30] | 0.0185 | 0.0925 |
| DGP+ K=2 | DGP reproduced | AUROC | +0.72 (5/5) | [+0.42, +1.02] | 0.0044 | 0.0351 |
| DGP+ K=2 | DGP reproduced | AUPRC | +5.41 (5/5) | [+3.32, +7.51] | 0.0089 | 0.0532 |

