"""Run upstream ``lerobot-train`` with held-out phase metrics in W&B."""

from __future__ import annotations

from typing import Any

from hepha_lerobot.policies.hepha_act_phase.evaluation_metrics import (
    consume_phase_eval_metrics,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_train import main as lerobot_train_main


def _install_phase_metric_logging() -> None:
    if getattr(WandBLogger.log_dict, "_hepha_phase_metrics", False):
        return
    original_log_dict = WandBLogger.log_dict

    def log_dict(
        self: WandBLogger,
        values: dict,
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> Any:
        if mode == "eval" and "eval_loss" in values:
            values = {**values, **consume_phase_eval_metrics()}
        return original_log_dict(
            self,
            values,
            step=step,
            mode=mode,
            custom_step_key=custom_step_key,
        )

    log_dict._hepha_phase_metrics = True  # type: ignore[attr-defined]
    WandBLogger.log_dict = log_dict


def main() -> None:
    _install_phase_metric_logging()
    lerobot_train_main()


if __name__ == "__main__":
    main()
