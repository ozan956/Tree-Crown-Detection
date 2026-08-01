# Density-map counter — test-set results

Modern learned baseline (CSRNet-style density regression) trained on
VHRTrees train, evaluated on the 222-image test set.

## Detection (peak-detected points, one-to-one matched, gap=10)
- precision 0.888, recall 0.834, F1 0.860
- tp=11301 fp=1422 fn=2251 (total GT 13552)

## Counting
- MAE 4.63 trees/image, RMSE 7.48

## Reference (this paper, one-to-one)
- WBF ensemble: P 0.968 R 0.878 F1 0.921
- Integrated: P 0.833 R 0.971 F1 0.897
- Integrated + gate (recovery-preserving): P 0.893 R 0.957 F1 0.924
