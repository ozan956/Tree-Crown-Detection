import pytest

@pytest.fixture
def two_gt_boxes():
    # two 10x10 boxes, corner form
    return [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)]

@pytest.fixture
def two_overlapping_boxes():
    # boxes share the region x,y in [5,10]
    return [(0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)]
