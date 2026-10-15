import numpy as np

from src.analytics.line_counter import LineCounter


def test_line_counter_direction():
    lc = LineCounter(start=(0, 100), end=(200, 100), name="test")
    for y in [50, 80, 120, 150]:
        lc.update(track_id=1, bbox=np.array([90, y, 110, y + 20]))
    assert lc.in_count == 1
    assert lc.out_count == 0


def test_line_counter_no_cross():
    lc = LineCounter(start=(0, 100), end=(200, 100), name="test")
    for y in [10, 20, 30]:
        lc.update(track_id=1, bbox=np.array([90, y, 110, y + 20]))
    assert lc.in_count == 0
    assert lc.out_count == 0