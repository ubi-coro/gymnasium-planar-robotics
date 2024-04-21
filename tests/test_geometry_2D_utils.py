import pytest
import numpy as np
from gymnasium_planar_robotics.utils import geometry_2D_utils


@pytest.mark.parametrize(
    'p1, p2, q1, q2, expected_result',
    [
        (np.array([[0, 0]]), np.array([[1, 1]]), np.array([[0, 0]]), np.array([[1, 0.5]]), np.array([True])),
        (
            np.array(
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                    [-2, 0],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                ]
            ),
            np.array(
                [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 0.8],
                    [-1, 0.8],
                    [1, 1],
                    [0, 0.5],
                    [1, 1],
                    [-1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [-1, -1],
                ]
            ),
            np.array(
                [
                    [0, 0],
                    [0.1, 0.5],
                    [1, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-2, 0],
                    [0, 1],
                    [1, 0],
                    [0, 0],
                    [0, 1],
                    [0.5, 0.5],
                    [-0.5, -0.5],
                ]
            ),
            np.array(
                [
                    [0.1, 0.5],
                    [1, 1],
                    [1, 1],
                    [1, 0.8],
                    [-1, 0.8],
                    [1, 1],
                    [1, 1],
                    [0, 0.5],
                    [1, 1],
                    [-1, 1],
                    [1, 1],
                    [0, 2],
                    [1, 2],
                    [1, 1],
                    [2, 2],
                    [-2, -2],
                ]
            ),
            np.array([True, True, True, True, True, True, True, False, False, False, False, True, True, True, True, True]),
        ),
    ],
)
def test_line_segments_intersect_check(p1, p2, q1, q2, expected_result):
    assert (expected_result == geometry_2D_utils.check_line_segments_intersect(p1=p1, p2=p2, q1=q1, q2=q2)).all()


@pytest.mark.parametrize(
    'qpos_r1, qpos_r2, size_r1, size_r2, expected_result',
    [
        (
            np.array([[0.05, 0.05, 0, 0.9238795, 0.0, 0.0, 0.3826834]]),
            np.array([[0.05, 0.05, 0, 1, 0.0, 0.0, 0]]),
            np.array([[0.08, 0.08]]),
            np.array([[0.08, 0.08]]),
            np.array([True]),
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                    [0.0, 0.0, 0, 1, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [-2 * 0.08, -2 * 0.08, 0, 1, 0, 0, 0],
                    [-2 * 0.08, 2 * 0.08, 0, 1, 0, 0, 0],
                    [2 * 0.08, 2 * 0.08, 0, 1, 0, 0, 0],
                    [2 * 0.08, -2 * 0.08, 0, 1, 0, 0, 0],
                    [-2 * 0.08, -2 * 0.08, 0, 1, 0, 0, 0],
                    [-2 * 0.08, 2 * 0.08, 0, 1, 0, 0, 0],
                    [-2 * 0.08, 2 * 0.08, 0, 1, 0, 0, 0],
                    [-0.08, -2 * 0.08, 0, 1, 0, 0, 0],
                    [2 * 0.08, -0.08, 0, 1, 0, 0, 0],
                    [-2 * 0.08, -0.08, 0, 1, 0, 0, 0],
                    [-np.sqrt(2) * 0.08 - 0.08, -0.08, 0, 0.9238795, 0.0, 0.0, 0.3826834],
                    [-np.sqrt(2) * 0.08 - 0.08, 0.08, 0, 0.9238795, 0.0, 0.0, 0.3826834],
                    [-np.sqrt(2) * 0.08 - 0.08, -0.04, 0, 0.9238795, 0.0, 0.0, 0.3826834],
                    [np.sqrt(2) * 0.08 + 0.08, 0.08, 0, 0.9238795, 0.0, 0.0, 0.3826834],
                    [np.sqrt(2) * 0.08 + 0.08, -0.08, 0, 0.9238795, 0.0, 0.0, 0.3826834],
                    [np.sqrt(2) * 0.08 + 0.08, -0.04, 0, 0.9238795, 0.0, 0.0, 0.3826834],
                ]
            ),
            np.repeat(np.array([[0.08, 0.08]]), 16, axis=0),
            np.repeat(np.array([[0.08, 0.08]]), 16, axis=0),
            np.repeat(np.array([True]), 16),
        ),
    ],
)
def test_rectangles_intersect_check(qpos_r1, qpos_r2, size_r1, size_r2, expected_result):
    assert (
        expected_result
        == geometry_2D_utils.check_rectangles_intersect(qpos_r1=qpos_r1, qpos_r2=qpos_r2, size_r1=size_r1, size_r2=size_r2)
    ).all()
