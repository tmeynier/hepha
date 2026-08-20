"""Run LeRobot training with discounted-return targets for Hepha ACT AWR."""

from __future__ import annotations

from hepha_lerobot.awr.dataset import AWRReturnDataset
from hepha_lerobot.policies.hepha_act_awr.evaluation_metrics import (
    consume_awr_eval_metrics,
)
from hepha_lerobot.policies.hepha_act_awr.processor_hepha_act_awr import (
    _batch_to_transition_with_return,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.scripts import lerobot_train


def _install_return_dataset_wrapper() -> None:
    original = lerobot_train.make_train_eval_datasets

    def make_train_eval_datasets(cfg):
        train_dataset, eval_dataset = original(cfg)
        discount = cfg.policy.awr_discount
        train_dataset = AWRReturnDataset(train_dataset, discount=discount)
        if eval_dataset is not None:
            eval_dataset = AWRReturnDataset(eval_dataset, discount=discount)
        return train_dataset, eval_dataset

    lerobot_train.make_train_eval_datasets = make_train_eval_datasets

    original_processors = lerobot_train.make_pre_post_processors

    def make_pre_post_processors(*args, **kwargs):
        preprocessor, postprocessor = original_processors(*args, **kwargs)
        preprocessor.to_transition = _batch_to_transition_with_return
        return preprocessor, postprocessor

    lerobot_train.make_pre_post_processors = make_pre_post_processors

    original_log_dict = WandBLogger.log_dict

    def log_dict(self, values, step=None, mode="train", custom_step_key=None):
        if mode == "eval" and "eval_loss" in values:
            values = {**values, **consume_awr_eval_metrics()}
        return original_log_dict(
            self,
            values,
            step=step,
            mode=mode,
            custom_step_key=custom_step_key,
        )

    WandBLogger.log_dict = log_dict


def main() -> None:
    _install_return_dataset_wrapper()
    lerobot_train.main()


if __name__ == "__main__":
    main()
