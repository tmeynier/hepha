from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hepha_lerobot.evaluation import physical_rollout
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _policy_config(
    *,
    state_shape=(12,),
    image_shape=(3, 256, 256),
    action_shape=(12,),
    policy_type="act",
    camera_name="head_camera",
):
    return SimpleNamespace(
        type=policy_type,
        max_state_dim=32,
        input_features={
            OBS_STATE: SimpleNamespace(shape=state_shape),
            f"{OBS_IMAGES}.{camera_name}": SimpleNamespace(shape=image_shape),
        },
        output_features={ACTION: SimpleNamespace(shape=action_shape)},
    )


def test_physical_policy_schema_matches_recorded_dataset() -> None:
    physical_rollout.validate_physical_policy_features(
        _policy_config(),
        camera_name="head_camera",
        width=256,
        height=256,
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_policy_config(state_shape=(15,)), "observation.state"),
        (_policy_config(image_shape=(3, 480, 640)), "head_camera"),
        (_policy_config(action_shape=(15,)), "action"),
        (_policy_config(policy_type="diffusion"), "ACT and PI0"),
    ],
)
def test_physical_policy_schema_rejects_incompatible_checkpoint(config, message) -> None:
    with pytest.raises(ValueError, match=message):
        physical_rollout.validate_physical_policy_features(
            config,
            camera_name="head_camera",
            width=256,
            height=256,
        )


def test_pi0_schema_accepts_padded_state_and_saved_camera_rename() -> None:
    config = _policy_config(
        state_shape=(32,),
        image_shape=(3, 224, 224),
        policy_type="pi0",
        camera_name="base_0_rgb",
    )
    preprocessor = SimpleNamespace(
        steps=[
            SimpleNamespace(
                rename_map={
                    f"{OBS_IMAGES}.head_camera": f"{OBS_IMAGES}.base_0_rgb",
                }
            )
        ]
    )

    physical_rollout.validate_physical_policy_features(
        config,
        preprocessor=preprocessor,
        camera_name="head_camera",
        width=256,
        height=256,
    )


def test_pi0_schema_rejects_missing_saved_camera_rename() -> None:
    config = _policy_config(
        state_shape=(32,),
        image_shape=(3, 224, 224),
        policy_type="pi0",
        camera_name="base_0_rgb",
    )

    with pytest.raises(ValueError, match="absent from the policy inputs"):
        physical_rollout.validate_physical_policy_features(
            config,
            camera_name="head_camera",
            width=256,
            height=256,
        )


def test_physical_observation_uses_canonical_joint_order_and_rgb() -> None:
    positions = {
        joint: float(index) for index, joint in enumerate(physical_rollout.CANONICAL_JOINTS)
    }
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    observation = physical_rollout.build_physical_policy_observation(
        positions,
        rgb,
        camera_name="head_camera",
    )

    assert observation[OBS_STATE].dtype == np.float32
    assert observation[OBS_STATE].tolist() == list(range(12))
    assert observation[f"{OBS_IMAGES}.head_camera"] is rgb


def test_physical_action_maps_canonical_joint_targets() -> None:
    values = torch.arange(12, dtype=torch.float32).unsqueeze(0)

    targets = physical_rollout.action_targets(values)

    assert list(targets) == list(physical_rollout.CANONICAL_JOINTS)
    assert targets["shoulder_r"] == 0.0
    assert targets["finger_l"] == 11.0


def test_physical_action_rejects_nonfinite_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match="action shape"):
        physical_rollout.action_targets(np.zeros(11, dtype=np.float32))
    invalid = np.zeros(12, dtype=np.float32)
    invalid[3] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        physical_rollout.action_targets(invalid)
