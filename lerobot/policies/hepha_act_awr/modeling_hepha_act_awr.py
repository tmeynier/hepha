"""LeRobot ACT with a shared observation-value head and AWR training loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from .configuration_hepha_act_awr import HephaActAWRConfig
from .evaluation_metrics import accumulate_awr_eval_metrics, reset_awr_eval_metrics
from .processor_hepha_act_awr import AWR_RETURN


class HephaActAWRPolicy(ACTPolicy):
    """Keep ACT's actor intact and attach a critic to its shared input encoder.

    The value head consumes ACT's projected camera, state and environment tokens.
    These features are upstream of the action decoder and VAE action target, so
    the critic cannot leak demonstrated future actions into its value estimate.
    """

    config_class = HephaActAWRConfig
    name = "hepha_act_awr"

    def __init__(
        self,
        config: HephaActAWRConfig,
        *,
        dataset_meta: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.config = config
        token_count = (
            len(config.image_features)
            + int(config.robot_state_feature is not None)
            + int(config.env_state_feature is not None)
        )
        if token_count == 0:
            raise ValueError("The AWR value network needs at least one observation input")
        self.value_head = nn.Sequential(
            nn.LayerNorm(token_count * config.dim_model),
            nn.Linear(token_count * config.dim_model, config.value_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.value_head_dropout),
            nn.Linear(config.value_hidden_dim, 1),
        )
        self._camera_tokens: list[Tensor] = []
        self._camera_hook = self.model.encoder_img_feat_input_proj.register_forward_hook(
            self._capture_camera_token
        )
        if dataset_meta is not None:
            features = getattr(dataset_meta, "features", {})
            if "next.reward" not in features:
                raise ValueError("hepha_act_awr requires a dataset with next.reward")

    def _capture_camera_token(
        self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        self._camera_tokens.append(output.mean(dim=(-2, -1)))

    def train(self, mode: bool = True) -> HephaActAWRPolicy:
        entering_evaluation = self.training and not mode
        policy = super().train(mode)
        if entering_evaluation:
            reset_awr_eval_metrics()
        return policy

    def _value(self, batch: dict[str, Tensor]) -> Tensor:
        tokens: list[Tensor] = []
        if self.config.robot_state_feature:
            tokens.append(self.model.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            tokens.append(self.model.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        tokens.extend(self._camera_tokens)
        expected = (
            len(self.config.image_features)
            + int(self.config.robot_state_feature is not None)
            + int(self.config.env_state_feature is not None)
        )
        if len(tokens) != expected:
            raise RuntimeError(f"Expected {expected} ACT value tokens, got {len(tokens)}")
        return self.value_head(torch.cat(tokens, dim=-1)).squeeze(-1)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if AWR_RETURN not in batch:
            raise KeyError(f"Training {self.name!r} requires {AWR_RETURN!r}")
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        self._camera_tokens.clear()
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
        value = self._value(batch)

        abs_error = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid = (~batch["action_is_pad"]).unsqueeze(-1)
        valid_count = valid.sum(dim=(1, 2)) * abs_error.shape[-1]
        actor_per_sample = (abs_error * valid).sum(dim=(1, 2)) / valid_count.clamp_min(1)
        kld_per_sample = torch.zeros_like(actor_per_sample)
        if self.config.use_vae and log_sigma_x2_hat is not None:
            kld_per_sample = (
                -0.5
                * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())
            ).sum(-1)
            actor_per_sample = actor_per_sample + self.config.kl_weight * kld_per_sample

        return_target = batch[AWR_RETURN].float().reshape(-1)
        if return_target.shape != value.shape:
            raise ValueError(
                f"AWR return shape {tuple(return_target.shape)} does not match "
                f"value shape {tuple(value.shape)}"
            )
        advantage = return_target - value.detach()
        weight_advantage = advantage
        if self.config.awr_normalize_advantage and advantage.numel() > 1:
            weight_advantage = (advantage - advantage.mean()) / advantage.std(
                unbiased=False
            ).clamp_min(1e-6)
        weights = torch.exp(weight_advantage / self.config.awr_beta).clamp(
            max=self.config.awr_max_weight
        )
        weights = weights / weights.mean().clamp_min(1e-6)

        actor_loss = (weights * actor_per_sample).mean()
        value_loss = F.mse_loss(value, return_target)
        loss = actor_loss + self.config.value_loss_weight * value_loss
        metrics = {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "value_mean": value.mean().item(),
            "return_mean": return_target.mean().item(),
            "advantage_mean": advantage.mean().item(),
            "awr_weight_mean": weights.mean().item(),
            "awr_weight_max": weights.max().item(),
            "l1_loss": (
                (abs_error * valid).sum() / valid.sum().mul(abs_error.shape[-1]).clamp_min(1)
            ).item(),
        }
        if self.config.use_vae:
            metrics["kld_loss"] = kld_per_sample.mean().item()
        if not self.training:
            accumulate_awr_eval_metrics(metrics, return_target.shape[0])
        return loss, metrics

    @torch.no_grad()
    def predict_value(self, batch: dict[str, Tensor]) -> Tensor:
        """Return V(s) without changing ACT's action queue."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        self._camera_tokens.clear()
        self.model(batch)
        return self._value(batch)
