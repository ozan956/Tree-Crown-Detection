"""Locks the paper's novel-core claim so it can't silently regress."""
from improvement.beyond_ceiling import coverage_counts, characterize


def test_integrated_exceeds_dl_union_ceiling():
    dl, integ, beyond, net, total = coverage_counts()
    assert total == 13552
    # integrated must cover MORE GT trees than the full detector union
    assert integ > dl
    # net gain is a real, non-trivial fraction (guards against a regression to ~0)
    assert net >= 400, net
    assert beyond >= net  # the beyond-ceiling set is at least the net gain


def test_beyond_ceiling_crowns_are_smaller():
    med, sizes = characterize()
    # the recovered crowns are systematically smaller than detected ones —
    # the mechanistic claim. Require a clear (>1.5x) size gap.
    ratio = med["dl_reachable"]["area"] / med["beyond_ceiling"]["area"]
    assert ratio > 1.5, ratio
    # and there are enough of them to characterize
    assert sizes["beyond_ceiling"] >= 300
