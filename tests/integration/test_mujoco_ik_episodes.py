from __future__ import annotations

import pytest

from simulation import SimulationConfig
from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.ik import MujocoIKController

FPS = 30


def _reproduction_command(seed: int, episode_seconds: float) -> str:
    return (
        ".venv/bin/hepha-record \\\n"
        "  --backend mujoco \\\n"
        "  --controller ik \\\n"
        "  --repo-id tmeynier/hepha_mujoco_ik \\\n"
        f"  --root datasets/hepha_mujoco_ik_seed_{seed} \\\n"
        "  --episodes 1 \\\n"
        f"  --episode-seconds {episode_seconds:g} \\\n"
        "  --max-attempts 1 \\\n"
        f"  --seed {seed} \\\n"
        "  --viewer true \\\n"
        "  --debug true \\\n"
        "  --overwrite"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_mujoco_ik_episode_succeeds_headlessly(
    ik_seed: int, pytestconfig: pytest.Config
) -> None:
    """Run one complete physical episode and identify any failing seed."""

    episode_seconds = pytestconfig.getoption("--ik-episode-seconds")
    if episode_seconds <= 0:
        raise pytest.UsageError("--ik-episode-seconds must be positive")
    maximum_frames = max(2, round(episode_seconds * FPS))

    with MujocoBackend(
        SimulationConfig(render=False, viewer=False, debug=False, fps=FPS)
    ) as backend:
        controller = MujocoIKController(backend, seed=ik_seed)
        controller.reset(episode_seed=0)
        initial_status = controller.status

        for frame_index in range(maximum_frames):
            progress = frame_index / (maximum_frames - 1)
            backend.send_action(controller.action(progress))
            backend.step()
            if controller.done:
                break

        result = controller.status if controller.done else "episode timed out"
        reproduction = _reproduction_command(ik_seed, episode_seconds)
        assert controller.done and controller.successful, (
            f"MuJoCo IK episode failed for seed {ik_seed}.\n"
            f"Initial state: {initial_status}\n"
            f"Result after {frame_index + 1}/{maximum_frames} frames: {result}\n"
            f"Reproduce with the viewer:\n{reproduction}"
        )
