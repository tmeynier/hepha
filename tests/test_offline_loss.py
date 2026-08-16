from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from hepha_lerobot.evaluation import offline_loss


class _EvalDataset:
    episodes: ClassVar[list[int]] = [7, 8]
    meta = SimpleNamespace(camera_keys=(), has_language_columns=False)

    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"action": torch.tensor([float(index)])}


class _Policy:
    def eval(self) -> None:
        pass

    def forward(self, batch: dict[str, torch.Tensor]):
        return batch["action"].mean(), {}


def test_eval_loss_matches_lerobot_batch_mean_aggregation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    (policy_path / "config.json").touch()
    (policy_path / "train_config.json").touch()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            root=dataset_root,
            repo_id="hepha/test",
            eval_split=0.1,
        ),
        policy=None,
        batch_size=2,
        max_eval_samples=0,
        persistent_workers=False,
        prefetch_factor=2,
        dataloader_multiprocessing_context=None,
    )
    monkeypatch.setattr(
        offline_loss.TrainPipelineConfig,
        "from_pretrained",
        staticmethod(lambda _path: cfg),
    )
    monkeypatch.setattr(
        offline_loss,
        "_load_policy_and_preprocessor",
        lambda _path, _device: (
            _Policy(),
            SimpleNamespace(use_amp=False),
            lambda batch: batch,
        ),
    )
    monkeypatch.setattr(
        offline_loss,
        "make_train_eval_datasets",
        lambda _cfg: (object(), _EvalDataset()),
    )

    report = offline_loss.compute_eval_loss(
        policy_path=policy_path,
        dataset_root=dataset_root,
        device="cpu",
        show_progress=False,
    )

    # LeRobot averages the three batch means: mean([0, 1]), mean([2, 3]), mean([4]).
    assert report.loss == pytest.approx((0.5 + 2.5 + 4.0) / 3.0)
    assert report.batches == 3
    assert report.samples == 5
    assert report.episodes == 2
    assert report.batch_size == 2
