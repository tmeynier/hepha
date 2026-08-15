from __future__ import annotations

import numpy as np
import pytest
from lerobot.utils.constants import OBS_ENV_STATE

from hepha_lerobot.conditioning import (
    DRAWER_CONDITION_NAMES,
    drawer_condition,
    drawer_condition_feature,
    drawer_task,
)


def test_drawer_condition_is_one_hot() -> None:
    condition = drawer_condition(5)

    np.testing.assert_array_equal(
        condition,
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    )


def test_drawer_condition_uses_native_lerobot_environment_state() -> None:
    feature = drawer_condition_feature()[OBS_ENV_STATE]

    assert feature["shape"] == (9,)
    assert feature["names"] == list(DRAWER_CONDITION_NAMES)


def test_drawer_task_identifies_the_requested_drawer() -> None:
    assert drawer_task("Move the cube to drawer {drawer_index}.", 3) == (
        "Move the cube to drawer 3."
    )


@pytest.mark.parametrize("drawer_index", [0, 10])
def test_drawer_condition_rejects_invalid_indices(drawer_index: int) -> None:
    with pytest.raises(ValueError, match="Drawer index"):
        drawer_condition(drawer_index)
