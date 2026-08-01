# Precision-gate improvement — held-out results

Image-disjoint split: gate trained on first 111 images, evaluated on last 111.
Metrics are one-to-one matched, gap=10, on the test half only.

| config | precision | recall | F1 |
|---|---|---|---|
| baseline (keep all far pts) | 0.828 | 0.973 | 0.894 |
| gated @P>=0.4 | 0.893 | 0.957 | 0.924 |
| gated @P>=0.5 | 0.918 | 0.940 | 0.929 |
| gated @P>=0.6 | 0.942 | 0.921 | 0.931 |
| gated @P>=0.7 | 0.962 | 0.908 | 0.934 |

Reference (full test set, Phase-1): integrated F1 0.897, WBF ensemble F1 0.921.
