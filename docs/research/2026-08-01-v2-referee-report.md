# Referee Report — article-v2.tex (compiled)

**Reviewed:** the assembled, compiled `article-v2.tex` (11-page PDF, read pages
1–6 as rendered). Written as a journal/applied-venue referee would, after the
Direction A/B/C work.

**Recommendation: ACCEPT WITH MINOR REVISIONS** for a solid applied /
remote-sensing venue (e.g. a good RS journal or applied-CV track). Not a
top-tier vision-conference paper — and it does not claim to be. The manuscript
is now internally consistent, honestly scoped, and the central contribution is
real and validated across two datasets.

---

## Summary of the submission

The paper integrates five deep detectors (via WBF) with a traditional
color+Gabor+watershed pipeline through a rule-based procedure, and asks what the
traditional side adds once a strong ensemble is in place. Its measured answer:
the traditional operator recovers 3.3% of trees beyond the entire deep-ensemble
ceiling; these are systematically small/faint crowns; an appearance gate turns
the recall gain into an operating point that dominates the ensemble; and the
mechanism replicates on an independent US airborne benchmark (NEON/DeepForest).

## Strengths

1. **A genuine, quantified contribution.** The beyond-ceiling recovery, measured
   against the union of six detectors at their floors, is a real result that the
   literature does not already report (verified via adversarial search). The
   mechanistic characterization (recovered crowns 2.44× smaller) elevates it from
   an anecdote to a finding.
2. **Cross-dataset validation done fairly.** Using DeepForest on NEON — a
   competent, on-domain detector — avoids the domain-shift confound that would
   have made the result trivial. Replication of the small-crown mechanism
   (1.91×) on a different continent/sensor is convincing.
3. **Honest, quantitative evaluation.** One-to-one matching with precision,
   recall, F1, and mAP corrects the recall-only scoring of the prior version.
   The recall-only 96% is reported transparently as a reference, not the headline.
4. **The gate-vs-novelty tension is handled, not hidden.** The paper openly
   shows the F1-max gate discards most recoveries and adopts the
   recovery-preserving point instead. Reviewers respect this.
5. **Reproducibility.** Public code that regenerates every number and figure.

## Required minor revisions

1. **NEON precision caveat must not be buried.** The 0.25 standalone precision
   of the traditional pipeline on NEON is stated (good), but a skimming reader
   could take the "94.2% union coverage" at face value. Add one sentence in the
   NEON subsection making explicit that the union figure is generous-match and
   that the *deployable* cross-dataset claim is the mechanism (1.91×), not a
   coverage gain.
2. **Figure 1 placement.** The coverage ladder is introduced as the "workflow"
   figure in Section 2 but it is a *results* figure; either add the actual
   traditional-pipeline workflow diagram (the v1 Figure 1) or move the ladder to
   Section 5 and retitle Section 2's reference. Currently the Section-2 text
   promises a workflow and shows results.
3. **Statistical treatment.** Report at least a bootstrap confidence interval on
   the headline F1 numbers (integrated+gate vs WBF), since the margin (0.924 vs
   0.921) is small. This is cheap to add and pre-empts a "within noise?" question
   on the primary comparison.
4. **Define the gate's features and training precisely** in the method (currently
   only in Results). One or two sentences: the 8 features, the logistic model,
   and the image-disjoint training protocol.

## Optional (would strengthen, not required)

- A per-biome recalibration of the traditional pipeline on NEON, with a fair
  post-recalibration F1, would convert the honest boundary into a positive
  result.
- The density-map baseline could be tuned harder; state that its F1 is a
  competent-but-not-exhaustive lower bound (already implied).

## Assessment of specific claims (referee spot-checks)

- **"+448 trees / 3.3pp beyond ceiling"** — sound; conservative net number, not
  the 692-set. Good.
- **"dominates the ensemble on the P–R frontier"** — supported at the
  recovery-preserving point (0.893/0.957 vs 0.968/0.878); the word "dominates" is
  defensible since the integrated point is not Pareto-dominated by WBF. Keep.
- **"mechanism replicates cross-dataset"** — supported (1.91× on NEON); the
  boundary (pipeline needs recalibration) is stated. Good.
- **"AP50 ≈ 0.85 for all detectors"** — consistent with the table. Good.

## Verdict

The manuscript makes a modest but real and honestly-bounded contribution, is
methodologically sound, is validated on two datasets, and is fully reproducible.
The required revisions are minor and mostly presentational. I would accept it
after those are addressed.
