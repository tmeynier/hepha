from __future__ import annotations

import torch
from hepha_lerobot.conditioning import NEXT_TASK_PHASE
from hepha_lerobot.policies.hepha_act_phase import (
    HephaActPhaseConfig,
    HephaActPhasePolicy,
)
from hepha_lerobot.policies.hepha_act_phase.evaluation_metrics import (
    consume_phase_eval_metrics,
)
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _small_policy() -> HephaActPhasePolicy:
    config = HephaActPhaseConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(14,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(15,)),
        },
        chunk_size=4,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        n_vae_encoder_layers=1,
        phase_head_dropout=0.0,
    )
    return HephaActPhasePolicy(config)


def _batch() -> dict[str, torch.Tensor]:
    environment_state = torch.zeros(2, 14)
    environment_state[:, 0] = 1.0
    environment_state[0, 9] = 1.0
    environment_state[1, 10] = 1.0
    return {
        OBS_STATE: torch.zeros(2, 15),
        OBS_ENV_STATE: environment_state,
        ACTION: torch.zeros(2, 4, 15),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        NEXT_TASK_PHASE: torch.tensor([[0], [2]], dtype=torch.int64),
    }


def test_phase_act_combines_act_and_phase_losses() -> None:
    policy = _small_policy()

    loss, metrics = policy.forward(_batch())
    loss.backward()

    assert loss.ndim == 0
    assert metrics.keys() >= {
        "act_loss",
        "l1_loss",
        "kld_loss",
        "phase_loss",
        "weighted_phase_loss",
        "phase_accuracy",
        "phase_transition_accuracy",
        "phase_hold_accuracy",
    }
    assert policy.phase_head[-1].weight.grad is not None


def test_phase_act_predicts_actions_and_phase_logits_together() -> None:
    policy = _small_policy().eval()
    batch = _batch()
    batch.pop(ACTION)
    batch.pop("action_is_pad")
    batch.pop(NEXT_TASK_PHASE)

    actions, phase_logits = policy.predict_action_chunk_with_phase(batch)

    assert actions.shape == (2, 4, 15)
    assert phase_logits.shape == (2, 5)


def test_phase_act_accumulates_held_out_phase_metrics() -> None:
    policy = _small_policy().eval()

    policy.forward(_batch())
    metrics = consume_phase_eval_metrics()

    assert metrics.keys() >= {
        "phase_loss",
        "phase_accuracy",
        "phase_transition_accuracy",
        "phase_hold_accuracy",
    }
    assert consume_phase_eval_metrics() == {}
