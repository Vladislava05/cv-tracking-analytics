import numpy as np
import pytest

from src.detection.model import Detection
from src.tracking.byte_tracker import ByteTracker, iou_batch
from src.tracking.kalman import KalmanBoxTracker


def make_det(bbox, score=0.9, cls=0):
    return Detection(
        bbox=np.array(bbox, dtype=np.float32),
        score=score,
        class_id=cls,
        class_name="person",
    )


def test_iou_identical():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert np.isclose(iou_batch(a, b)[0, 0], 1.0)


def test_iou_no_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[100, 100, 110, 110]], dtype=np.float32)
    assert iou_batch(a, b)[0, 0] == 0.0


def test_kalman_predict_update():
    trk = KalmanBoxTracker(np.array([0, 0, 10, 10], dtype=np.float32))
    trk.predict()
    trk.update(np.array([5, 0, 15, 10], dtype=np.float32))
    state = trk.get_state()
    assert state.shape == (4,)
    assert abs(state[0] - 5) < 3


def test_tracker_stable_id():
    tracker = ByteTracker(min_hits=1)
    prev_id = None
    for i in range(10):
        det = make_det([i * 5, 100, i * 5 + 30, 200])
        tracks = tracker.update([det])
        assert len(tracks) == 1
        if prev_id is not None:
            assert tracks[0].id == prev_id
        prev_id = tracks[0].id


def test_tracker_two_objects():
    tracker = ByteTracker(min_hits=1)
    for i in range(5):
        det1 = make_det([i * 5, 100, i * 5 + 30, 200])
        det2 = make_det([500 + i * 5, 100, 530 + i * 5, 200])
        tracks = tracker.update([det1, det2])
        assert len(tracks) == 2