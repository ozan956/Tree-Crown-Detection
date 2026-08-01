# Novelty Assessment & Improvement Roadmap — Tree Crown Detection

**Date:** 2026-08-01
**Paper:** arXiv:2507.01502 — "Integrating Traditional and Deep Learning Methods to Detect Tree Crowns in Satellite Images" (preprint, revisable)
**Scope:** After Phase 1 (rigorous re-evaluation), this document answers three questions the author asked: (1) how novel is the approach, (2) are we satisfied with the result, (3) how do we improve — grounded in a literature scan and a visual diagnosis of the actual results.

---

## 1. Are we satisfied with the current result? — Verdict: **partially, and now honestly so**

Phase 1 replaced the paper's single recall number with a defensible picture. The honest scorecard (one-to-one matching, score≥0.8):

| Method | Recall | Precision | F1 | AP50 |
|---|---|---|---|---|
| WBF ensemble | 0.878 | **0.968** | **0.921** | 0.864 |
| Rule-based integrated | **0.971** | 0.833 | 0.897 | n/a |

**What's genuinely strong (keep and lead with it):**
- The integrated method reaches **recall 0.971, beyond the WBF ensemble's maximum achievable recall of 0.926 at *any* score threshold** (see `results/pr_curve.png`). Verified apples-to-apples: even scoring WBF with the same lenient center-in-box criterion, WBF tops out at 92.5% vs the integrated 96.0%. **This is the paper's real, defensible contribution:** the traditional watershed/local-maxima step recovers trees the deep detectors never propose as boxes. A threshold sweep cannot replicate it.

**What we should NOT be satisfied with:**
- The integrated method's **F1 (0.897) is below the WBF ensemble's (0.921).** It buys +9.3 pts recall with −13.5 pts precision. That is a *recall-maximizing operating point*, not a uniform improvement — and the original paper hid this by never reporting precision.
- **2,647 false positives** (vs 388 false negatives). Precision is the weak axis, and it lives entirely in the traditional post-processing step (the DL detectors are 97–99% precise on their own).

**Bottom line:** the *idea* works and is now provable; the *execution* leaves a large, fixable precision gap. We should be satisfied with the reframing and dissatisfied with the precision — which is exactly where the improvement work should go.

---

## 2. Novelty assessment (literature scan, 2023–2026)

**The field has moved in three directions since the paper's references:**

1. **Foundation models / SAM + prompting.** SAM and Grounding-DINO+SAM are now common for crown *delineation* (e.g. desert-vegetation fine segmentation 2025; multi-sensor urban crown segmentation with NDVI-watershed+SAM, Sharifi 2025). Weakly-supervised, point/box-prompted segmentation is the trend.
2. **Density-map / crowd-counting for dense small trees.** Directly targets this paper's stated weakness (small clustered trees). Konttila 2025 ("counting trees … density mapping, submeter satellite"), Gominski et al. "density-map prediction network for individual tree detection," and LiDAR density-peak clustering (Fu 2024). These predict a continuous density surface and integrate it — far more robust to clustering than box detectors + watershed.
3. **Large-scale delineation models.** DeepTrees (Germany, ~218.7M crowns, 2025), the 2024 IEEE TGRS review (Zheng et al.) formalizing ITCD = detection + delineation.

**Where this paper sits:**
- **Distinctive:** an explicit *rule-based* integration layer that (a) uses WBF to fuse 5 detectors, then (b) validates/recovers tree centers via traditional color+Gabor+watershed+neighbor rules. Most hybrid papers apply traditional methods *sequentially* (detect→watershed-split) or as pure post-processing. A codified rule set that both *validates* DL boxes and *adds* traditional-only detections is relatively uncommon. **The "recovers trees beyond the detector ceiling" result (§1) is a real, publishable novelty hook** — but only now that we can prove it with the PR-curve.
- **Dated / vulnerable:** the "traditional features" are thin (fixed HSV threshold called "color invariants"; a single Gabor kernel called a "bank"). The comparison detectors (Faster-RCNN, YOLOv3, DETR, Swin-MaskRCNN) are 2018–2021. No SAM, no density-map baseline, no DeepForest comparison. A 2026 reviewer will ask "why not SAM / density maps?"

**Novelty verdict:** the *integration concept + the beyond-ceiling recall finding* is novel enough for a solid applied/remote-sensing venue **if** (a) the precision gap is addressed, (b) the "traditional features" claims are made truthful, and (c) at least one modern baseline (density-map or SAM) is added for context.

---

## 3. Visual diagnosis of the errors (from `resultant-new-version-all/` images)

Three failure modes, each seen directly and quantified:

- **FP-1 — crown over-splitting (dominant).** The watershed distance-transform produces multiple local maxima on one large crown → 3–4 "trees" where there is one. Seen clearly in `y8_1261` (id 61: 25 GT → 75 detections). Cause: `sure_fg` threshold `0.01*max` + `maximum_filter(size=5)` are far too permissive; every bump becomes a peak.
- **FP-2 — non-tree firing.** Detections land on grass, shrubs, bare soil, and rooftops. Seen in `st3_804` (id 208: 42 GT → 102 detections, dots on empty soil). Cause: fixed HSV green range + single Gabor gives no real tree-vs-vegetation discrimination; the joint map passes low grass.
- **FN — crown under-segmentation.** Touching crowns merge into one contour + one center. Seen in `a16_05_03` (id 203: 106 GT, 17 merged). Cause: watershed markers not separating adjacent peaks in dense canopy.

FP rate is high in **both** sparse (10.1/img) and dense (13.4/img) terrain → the precision problem is intrinsic to the **traditional pipeline**, not the detectors. This is the single highest-leverage target.

---

## 4. Improvement roadmap (ranked by leverage × effort)

### Tier A — highest leverage, runnable now on saved data (no GPU, no retraining)

**A1. Fix crown over-splitting (attacks FP-1, the biggest FP source).**
- Replace the naive `maximum_filter(size=5)` + `0.01*max` foreground threshold with **peak detection scaled to expected crown size**: `skimage.feature.peak_local_max` with `min_distance` set from the median GT crown radius (we have `avg_width/avg_height` already), and a distance-transform threshold tied to crown scale, not a flat 1%.
- Add a **merge rule**: local maxima closer than ~0.6× median crown diameter collapse to one (the code already has `merge_close_local_maxima` with a fixed 25px — make it crown-scale-adaptive).
- **Expected:** large FP reduction on sparse/olive terrain with little recall cost. Measurable immediately via the Phase-1 harness (`evaluate_all`).

**A2. Suppress non-tree firing (attacks FP-2).**
- The DL detectors are 97–99% precise. Use them as a **precision gate**: only *keep* a traditional-only detection if it has supporting evidence (texture contrast above local grass baseline, or crown-sized dark blob), and be stricter where NO detector box is nearby. Currently the neighbor rule (`required_nearby`) is loose (3–5 within 180px) and *adds* points liberally.
- Add a **local-contrast / NDVI-proxy test** (trees are darker/greener than surrounding dry grass in this Mediterranean set) before accepting a traditional-only center.
- **Expected:** cuts the grass/soil/rooftop FPs; directly raises the 0.833 precision.

**A3. Operating-point / trade-off reporting.**
- Report the integrated method at **multiple thresholds** and pick the F1-optimal one, not just recall-max. Add the integrated method's own precision–recall behavior as we vary its acceptance rules (a small sweep), so the paper can present "recall-max" and "F1-max" configurations. Turns the current single point into a curve — strictly stronger.

### Tier B — medium leverage, modest new code (still no retraining)

**B1. Add a density-map baseline (addresses novelty gap + small-tree weakness).**
- Implement or borrow a lightweight density-map counter as a *comparison* and potentially as a *fusion input* for dense clusters. This is the technique the 2025 literature uses for exactly this problem. Even as a baseline it answers the "why not modern methods?" reviewer question.

**B2. Truthful features / method upgrade (fixes the overclaim, may help precision).**
- Implement a real multi-orientation, multi-scale **Gabor bank** (the paper cites Jain–Farrokhnia but ships one kernel) and a genuine illumination-invariant color feature (e.g. normalized rgb or a vegetation index), replacing the fixed HSV threshold. Either make the text match the code, or make the code match the text — the latter may also improve FP-2.

### Tier C — higher effort, strongest paper (needs GPU/retraining)

**C1. SAM-based delineation for the "recovered" trees.** Use the traditional-only centers as **point prompts to SAM** to get accurate crown masks instead of watershed contours. Modernizes the delineation, aligns with 2025 SOTA, and likely improves boundary quality.

**C2. Retrain/add a modern detector** (e.g. DINO, Co-DETR, or a YOLOv8/v11) to refresh the 2018–2021 detector lineup, and re-run WBF. Raises the ensemble ceiling the integration builds on.

---

## 5. Recommended next step

Do **Tier A** first — it is fully runnable on the data already vendored, needs no GPU, and directly attacks the 2,647-FP precision gap that is the result's main weakness. Each change is measurable in minutes via the Phase-1 `evaluation/` harness, so we can report a before/after precision/F1 gain with evidence. That alone could lift the integrated F1 above the WBF ensemble's 0.921 — turning "recall-maximizing trade-off" into "uniform improvement," which is a materially stronger claim for the revision.

Tier B (density-map baseline + truthful features) is what makes the paper *current* for a 2026 venue. Tier C is the ambitious version.

**Files that would change (Tier A):** the traditional pipeline in `Traditional Methods/trad-methods.py` and `Integration/integration.py` (peak detection, merge rule, acceptance gate), measured by the existing `evaluation/` module. No changes to the evaluation code itself — it becomes the scoreboard.
