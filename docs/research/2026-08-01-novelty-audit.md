# Novelty Audit — What We Can and Cannot Claim

**Date:** 2026-08-01
**Purpose:** Honestly separate our genuinely novel contributions from competent
applications of established techniques, so the paper's claims survive review.
Based on a targeted literature scan (Google Scholar, 2010–2026) specifically
hunting for prior art that would undercut each claim.

## Method of this audit

For each candidate contribution we searched for the *strongest possible prior
art that would refute novelty*, not for support. A claim only survives if the
adversarial search failed to find established equivalents.

---

## Claims that are NOT novel (established — do not claim as contributions)

These are competent engineering we use, but the literature already owns them.
Presenting them as novel would invite rejection.

| Technique we use | Prior art found | Verdict |
|---|---|---|
| **Learned false-positive filter / gate on detector outputs** | Cheng et al. 2018 (Decoupled Classification Refinement — "hard false positive suppression"); Gao 2020/2022 (post-processing schemes for aerial detection); Wedegedara 2026 (learned filter, 55%→90% precision). | **Established.** Our precision gate is a solid application, not a new idea. |
| **Vegetation-index / appearance discrimination of trees vs grass/shrub/soil** | Ardila 2012 (false positives in gardens); O'Neil-Dunne 2014 (NDVI grass/shrub class); El Hoummaidi 2021; Martinoli 2026 (NDVI vs hedges/grass). | **Established.** Using color/texture/VI to reject non-tree detections is standard. |
| **Weighted Boxes Fusion of multiple detectors** | Solovyev 2021 (WBF); the authors' own prior work [10]. | **Established / prior self-work.** |
| **Watershed + deep learning hybrid** | Sharifi 2025 (NDVI-watershed+SAM); Moshkenani 2025 (YOLOv5+watershed); Tian 2025 (fusion module). | **Common.** Sequential detect→segment hybrids are widespread. |

**Consequence for the precision gate (our Tier-A result):** it is a legitimate
*improvement to our system* (F1 0.894→0.934) and worth reporting as an
engineering result, but it is **not a novel method**. Frame it as "we apply a
standard learned appearance filter to remove the non-tree firings," not as a
contribution. Its value is that it makes the system's operating point dominate
the ensemble — a result, not an invention.

---

## The ONE claim that survives the adversarial search (the real novel core)

**Claim:** *A traditional (non-learned) detection step recovers true tree crowns
that lie beyond the detection ceiling of a strong multi-detector deep-learning
ensemble — i.e. trees that no detector proposes as a box at any confidence
threshold — and we quantify this complementarity on a labelled benchmark.*

**Why it survives:** the targeted searches for this exact combination returned:
- "None of the visible abstracts describe a hybrid approach pairing ensemble
  methods with conventional techniques to recover objects beyond ensemble
  thresholds."
- "None explicitly quantify instances where watershed/template-matching detect
  trees that CNNs missed … this represents an open research opportunity rather
  than an established, quantified finding."

Papers acknowledge hybrids are *potentially* beneficial, but the **quantified,
benchmark-backed complementarity claim is under-documented.**

**Our evidence (rigorous version, verified on the data):**
- Best single detector reaches ≤ 86.9% of GT trees even at its most permissive
  threshold (generous center-in-box match).
- The WBF ensemble reaches 92.5%.
- The **union of all 6 detectors at their lowest thresholds** — the absolute
  theoretical ceiling of what the DL stack can propose — reaches **93.3%**.
- The integrated method (with the traditional watershed/local-maxima recovery)
  reaches **96.0%**.
- Therefore **362 GT trees (2.7% of the benchmark) are recovered that no
  detector proposes as a box at any confidence** — attributable purely to the
  traditional step.

This is the number the paper should be built around. It is defensible,
reproducible, and it directly answers the skeptic's objection ("just lower the
detector threshold") — lowering every detector to zero and unioning them still
leaves those 362 trees undetected.

---

## Honest positioning of the paper's contributions

1. **Primary (novel):** Quantified complementarity — traditional methods recover
   2.7% of trees beyond the full deep-ensemble ceiling on a 13,552-tree
   benchmark. First systematic, benchmark-backed measurement of this effect for
   tree crowns. *This is the paper's reason to exist.*
2. **Secondary (novel-ish, methodological):** A *rule-based integration
   protocol* that decides per-candidate whether to trust the detector, a
   neighbor, or the traditional evidence. The individual rules are not novel;
   the specific codified decision procedure and its evaluation are a modest
   methodological contribution — claim it modestly.
3. **Engineering (not novel, but strengthens the system):** The precision gate
   removes non-tree firings so the integrated operating point dominates the
   ensemble on the P–R frontier (F1 0.894→0.934). Report as a result, cite the
   standard techniques it builds on.
4. **Meta-contribution (valuable, honest):** A rigorous re-evaluation protocol
   (one-to-one matching, precision + recall + F1 + mAP, PR-curve vs
   operating-point analysis) that corrects the recall-only metric common in this
   sub-area. Reviewers respect this.

---

## What would make the novel core bulletproof (before claiming it)

The 2.7%-beyond-ceiling claim rests on one dataset (Marmara + İzmir, Turkey).
To make it solid rather than suggestive:
1. **Characterize the 362 recovered trees** — are they systematically smaller,
   in shadow, spectrally atypical? If we can show *why* detectors miss them and
   the traditional step catches them, the claim becomes mechanistic, not
   anecdotal. (Runnable now on our data — Tier A.)
2. **Rule out annotation noise** — confirm the 362 are real trees in the imagery,
   not GT boxes on non-trees. (Spot-check via the images.)
3. **Ideally, a second dataset** — even a small one — to show the complementarity
   isn't dataset-specific. (Future work; be explicit if not done.)

---

## Recommendation

**Do NOT** lead with the precision gate or the hybrid pipeline as novel — they
are established. **DO** lead with the quantified beyond-ceiling complementarity
(362 trees / 2.7%), make it mechanistic by characterizing the recovered trees,
and present everything else (gate, integration rules, re-evaluation) as
supporting engineering and methodology. This is a defensible, honest paper.

Next build step that serves the *novel* core: **characterize the 362
beyond-ceiling recovered trees** (size/spectral/context analysis) — this is what
turns "we found a 2.7% gap" into "here is what deep detectors systematically
miss and why traditional cues catch it." That is runnable now on the vendored
data.
