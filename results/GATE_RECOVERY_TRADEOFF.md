# Consistency check: does the precision gate destroy the novel beyond-ceiling recovery?

**Date:** 2026-08-01
**Why this matters:** The paper has two contributions that could conflict. The
novel core is that the traditional step recovers 692 crowns *beyond the deep
ensemble's ceiling* — and those recoveries are exactly the *far-from-box*
points. The precision gate improves F1 by *removing* far-from-box points. A
reviewer will immediately ask: **does the F1-optimal gate throw away the novel
recoveries?** We answer it explicitly.

## Result: the gate threshold is a recovery-vs-precision dial

Fraction of the 692 beyond-ceiling recoveries that survive the gate, by
threshold (in-sample gate for this diagnostic; the F1 numbers are the held-out
values from `IMPROVEMENT_RESULTS.md`):

| gate threshold | beyond-ceiling retained | integrated F1 (held-out) |
|---|---|---|
| 0.3 | 680/692 = **98%** | 0.922 |
| 0.4 | 652/692 = **94%** | 0.924 |
| 0.5 | 540/692 = 78% | 0.929 |
| 0.6 | 384/692 = 55% | 0.931 |
| 0.7 | 262/692 = 38% | 0.934 |

## Interpretation and recommended operating point

There is a real tension: the **F1-maximising** gate (0.7, F1 0.934) retains only
38% of the beyond-ceiling recoveries — it wins F1 largely by discarding the very
detections that are the paper's novelty. Reporting only F1=0.934 would
undercut the novel claim.

**Recommended operating point: gate = 0.4.** It retains **94%** of the
beyond-ceiling recoveries (the novel finding is preserved) while still achieving
**F1 0.924 > WBF ensemble 0.921** (the integration still dominates the
ensemble). This resolves the tension: we do not have to choose between the
novelty and beating the ensemble.

This is the honest framing for the paper: present the gate as a **tunable
recovery/precision dial**, report the full curve, and adopt the
recovery-preserving operating point (0.4) as the primary configuration —
noting that if maximum F1 is the sole objective, 0.7 is available at the cost of
recovery. Do NOT headline 0.934 without this caveat.

## Effect on the v2 draft

`docs/paper-v2/v2-sections-draft.tex` currently reports the gate at 0.962/0.908/
0.934 (the 0.7 point). It should be updated to lead with the **0.4 operating
point** (precision ~0.893, recall ~0.957, F1 0.924) as primary, present 0.934 as
the F1-max extreme, and state the recovery-retention trade explicitly so the
gate and the beyond-ceiling contribution are visibly consistent.
