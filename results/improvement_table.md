# Precision-gate improvement — held-out results

Gate: HistGradientBoosting on 10 crop features (8 appearance + dist-to-WBF
+ blob compactness); far-from-box points only; near-box points kept.
+NMS merges kept points closer than 26 px (over-split duplicates).
Image-disjoint split: trained on first 111 images, evaluated on last 111,
one-to-one matched, gap=10.

| config | precision | recall | F1 |
|---|---|---|---|
| integrated (no gate) | 0.828 | 0.973 | 0.894 |
| gate @0.4 | 0.928 | 0.955 | 0.942 |
| gate @0.4 + NMS | 0.932 | 0.954 | 0.943 |
| gate @0.5 | 0.937 | 0.950 | 0.943 |
| gate @0.5 + NMS | 0.941 | 0.949 | 0.945 |

Reference: WBF ensemble F1 0.921 (Phase-1).
