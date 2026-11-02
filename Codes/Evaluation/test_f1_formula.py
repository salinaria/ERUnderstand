"""Check the published F1 formula against f1score.compute_f1."""

from f1score import compute_f1


def test_extra_and_miss_without_misconnection():
    """GT 10, one extra, one miss, no misconnection: precision and recall are 0.9."""
    # TP=9, FP=1, FN=1. Recall uses GT, which equals TP+FN when C=0.
    assert compute_f1(10, 1, 1, 0) == 0.9


def test_misconnection_recall_uses_ground_truth():
    """Two misconnections are half FP and half FN, and recall still divides by GT.

    precision = 10/11, recall = 1, F1 = 20/21 = 0.952.
    Putting the half-misconnection into the recall denominator would give 0.909.
    """
    assert compute_f1(10, 0, 0, 2) == 0.952


def test_empty_ground_truth_is_zero():
    """A category with no ground-truth support is not a positive F1."""
    assert compute_f1(0, 1, 0, 0) == 0.0
