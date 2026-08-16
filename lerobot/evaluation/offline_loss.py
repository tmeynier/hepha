"""Recompute LeRobot's held-out policy loss from a saved training run."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hepha_lerobot.training.train import resolve_device
from lerobot.configs import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.utils.collate import lerobot_collate_fn
from tqdm import tqdm

from lerobot.policies import get_policy_class, make_pre_post_processors


@dataclass(frozen=True)
class EvalLossReport:
    loss: float
    batches: int
    samples: int
    episodes: int
    batch_size: int
    eval_split: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy_path",
        type=Path,
        help="Local policy directory containing model and train_config.json",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Local LeRobot dataset root. By default, use the saved path when it "
            "exists, otherwise datasets/<saved repo name>."
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Override the dataset repo ID stored in train_config.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the saved training batch size (required for exact comparison)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; zero is the safest local default",
    )
    return parser.parse_args()


def _resolve_policy_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file() and (path / "train_config.json").is_file():
        return path
    pretrained_model = path / "pretrained_model"
    if (pretrained_model / "config.json").is_file() and (
        pretrained_model / "train_config.json"
    ).is_file():
        return pretrained_model
    raise FileNotFoundError(
        f"Expected config.json and train_config.json in {path} (or its pretrained_model directory)"
    )


def _resolve_dataset_root(cfg: TrainPipelineConfig, root: Path | None) -> Path:
    if root is not None:
        resolved = root.expanduser().resolve()
    else:
        saved_root = Path(cfg.dataset.root).expanduser()
        if saved_root.is_dir():
            resolved = saved_root.resolve()
        else:
            repo_name = str(cfg.dataset.repo_id).rsplit("/", maxsplit=1)[-1]
            resolved = (Path("datasets") / repo_name).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"LeRobot dataset root not found: {resolved}. Pass its local path with --root."
        )
    return resolved


def _load_policy_and_preprocessor(policy_path: Path, device: str):
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.device = device
    policy_config.pretrained_path = policy_path
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(policy_path, config=policy_config)
    policy.to(device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(policy_path),
        pretrained_revision=policy_config.pretrained_revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    preprocessor.reset()
    return policy, policy_config, preprocessor


def _limit_eval_samples(eval_dataset: Any, max_eval_samples: int):
    """Match LeRobot's task-balanced max_eval_samples implementation."""

    if max_eval_samples <= 0 or not hasattr(eval_dataset, "hf_dataset"):
        return eval_dataset
    task_array = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
    unique_tasks = sorted(set(task_array.tolist()))
    per_task = max(1, max_eval_samples // len(unique_tasks))
    selected: list[int] = []
    for task_index in unique_tasks:
        frames = (task_array == task_index).nonzero()[0][:per_task]
        selected.extend(frames.tolist())
    return torch.utils.data.Subset(eval_dataset, selected)


def _worker_options(cfg: TrainPipelineConfig, num_workers: int) -> dict[str, Any]:
    if num_workers == 0:
        return {}
    options: dict[str, Any] = {
        "persistent_workers": bool(cfg.persistent_workers),
        "prefetch_factor": cfg.prefetch_factor,
    }
    if cfg.dataloader_multiprocessing_context is not None:
        options["multiprocessing_context"] = cfg.dataloader_multiprocessing_context
    return options


def compute_eval_loss(
    *,
    policy_path: Path,
    dataset_root: Path | None = None,
    repo_id: str | None = None,
    device: str = "auto",
    batch_size: int | None = None,
    num_workers: int = 0,
    show_progress: bool = True,
) -> EvalLossReport:
    """Reproduce the training loop's ``eval/eval_loss`` calculation locally."""

    policy_path = _resolve_policy_directory(policy_path)
    device = resolve_device(device)
    cfg = TrainPipelineConfig.from_pretrained(policy_path)
    cfg.dataset.root = _resolve_dataset_root(cfg, dataset_root)
    if repo_id is not None:
        cfg.dataset.repo_id = repo_id
    if cfg.dataset.eval_split <= 0.0:
        raise ValueError(
            "The saved train_config.json has no held-out split (dataset.eval_split <= 0)."
        )
    if batch_size is None:
        batch_size = cfg.batch_size
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    policy, policy_config, preprocessor = _load_policy_and_preprocessor(
        policy_path, device
    )
    # The checkpoint config is authoritative for ACT delta indices and feature
    # layout; the train config remains authoritative for the episode split.
    cfg.policy = policy_config
    _, eval_dataset = make_train_eval_datasets(cfg)
    if eval_dataset is None:
        raise RuntimeError("LeRobot did not create a held-out evaluation dataset")

    evaluated_dataset = _limit_eval_samples(eval_dataset, cfg.max_eval_samples)
    collate_fn = lerobot_collate_fn if eval_dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        evaluated_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        **_worker_options(cfg, num_workers),
    )

    use_amp = bool(getattr(policy_config, "use_amp", False))
    device_type = torch.device(device).type
    autocast = (
        torch.autocast(device_type=device_type) if use_amp else contextlib.nullcontext()
    )
    loss_sum = 0.0
    batch_count = 0
    sample_count = 0
    policy.eval()
    with torch.no_grad(), autocast:
        for batch in tqdm(
            dataloader,
            desc="Held-out evaluation",
            unit="batch",
            disable=not show_progress,
        ):
            for camera_key in eval_dataset.meta.camera_keys:
                if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                    batch[camera_key] = batch[camera_key].to(torch.float32) / 255.0
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            # This intentionally averages batch means, exactly like lerobot-train
            # 0.6.1 and therefore like the value logged to W&B.
            loss_sum += loss.item()
            batch_count += 1
            sample_count += int(batch["action"].shape[0])

    if batch_count == 0:
        raise RuntimeError("The held-out evaluation split contains no batches")
    return EvalLossReport(
        loss=loss_sum / batch_count,
        batches=batch_count,
        samples=sample_count,
        episodes=len(eval_dataset.episodes or []),
        batch_size=batch_size,
        eval_split=float(cfg.dataset.eval_split),
    )


def main() -> None:
    args = parse_args()
    report = compute_eval_loss(
        policy_path=args.policy_path,
        dataset_root=args.root,
        repo_id=args.repo_id,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Held-out split: {report.episodes} episodes, {report.samples} samples")
    print(
        f"Aggregation: {report.batches} batch means at batch_size={report.batch_size} "
        f"(eval_split={report.eval_split:g})"
    )
    print(f"eval/eval_loss: {report.loss:.9f}")


if __name__ == "__main__":
    main()
