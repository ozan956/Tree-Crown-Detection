# Final Referee Report — reworked standalone manuscript (article-v2.tex)

**Reviewed:** the reworked, compiled manuscript (11-page PDF) after: standalone
reframing (no revision scaffolding), new workflow figure, de-titled figures with
rich captions, and the bootstrap-CI correction. Read all pages as rendered.

**Recommendation: ACCEPT WITH MINOR REVISIONS** at a solid applied /
remote-sensing venue. The paper reads as a single coherent proposal, its central
claim is real and cross-validated, and — importantly — an earlier overclaim
(F1 "dominance") was caught by the requested CI analysis and corrected. The
remaining issues are small.

## What improved since the prior review (all prior required fixes addressed)

1. **Standalone framing ✓** — no "conference version / re-evaluation / prior
   version" language; the method and evaluation read as the original design.
2. **Statistical treatment ✓** — bootstrap CIs added; and they did real work:
   they showed integrated+gate F1 is *tied* with WBF, so the paper now claims
   "matches F1 while detecting substantially more trees (recall +5.6..+9.0pp,
   100% of resamples)" instead of "dominates." This is more honest and still a
   real contribution.
3. **Workflow figure ✓** — new Fig 1 matches the faithful features; the coverage
   ladder correctly moved to Results.
4. **Gate defined in Methods ✓**; **NEON precision-0.25 caveat surfaced ✓** with
   an explicit "deployable claim is the mechanism, not a coverage gain."
5. **Figures cleaned ✓** — titles/in-plot text removed, explanation in captions.

## Strengths

- The beyond-ceiling result (+448 trees no detector proposes; recovered crowns
  2.44× smaller) is quantified, mechanistic, and now cross-validated on NEON
  with a fair (on-domain DeepForest) detector.
- Honest scoping throughout: F1 tie stated plainly, NEON pipeline boundary
  stated, single-dataset magnitude caveated.
- Fully reproducible; every number traces to committed code.

## Remaining minor revisions (non-blocking)

1. **Abstract wording vs. CI finding.** The abstract says the gate "matches the
   ensemble's F1 while detecting substantially more trees" — good — but a
   skeptical reader will note the integrated method's *own* F1 (0.897 ungated,
   0.924 gated) is numerically at/below WBF's 0.921 in Table 1, while recall is
   far higher. Consider one clause in the abstract making explicit that the
   advantage is recall at matched F1, not higher F1, so the table and abstract
   are unmistakably consistent.
2. **Table 1 "best" bolding.** F1 0.924 (integrated+gate) is bolded as best, but
   per the CI it is statistically tied with WBF 0.921. Either bold both or
   neither, or add a footnote that the two are within the bootstrap CI. Bolding
   0.924 alone slightly overstates a tie.
3. **Scene-proxy honesty.** The scene-holdout uses filename prefixes as a
   region proxy; the caption/text should keep the word "proxy" (it does in
   §5.5) — ensure the limitation that these are not confirmed climate labels is
   retained in the Limitations list too (currently only implied).
4. **Density-map single-run.** Note it is one training run; a sentence that its
   F1 is a competent lower bound (not exhaustively tuned) pre-empts "you
   under-trained the baseline."

## Spot-checks (referee verification)

- Table 1 numbers match the results files and the figures. ✓
- "union 94.2% / integrated 97.5% / +448 / 3.3pp" internally consistent. ✓
- Bootstrap CI claim ([-0.019,+0.007] F1; +5.6..+9.0pp recall) matches the
  computed values. ✓
- NEON "1.91× smaller, DeepForest 78.4%" matches the cross-dataset results. ✓
- All 30 citations resolve; 8 figures present; compiles with 0 undefined refs. ✓

## Verdict

A modest but genuine, honestly-bounded, well-evidenced and reproducible
contribution, now free of the F1-dominance overclaim. The four minor revisions
are presentational. Accept after they are addressed.
