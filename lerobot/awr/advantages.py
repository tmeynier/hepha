"""Freeze a fitted value checkpoint and materialize fixed AWR advantages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hepha_lerobot.awr.dataset import discounted_return_map
from hepha_lerobot.evaluation.conditioned_rollout import _load_policy
from hepha_lerobot.training.train import resolve_device
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("value_policy", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if not args.value_policy.is_dir():
        raise FileNotFoundError(f"Value checkpoint not found: {args.value_policy}")
    if not args.root.is_dir():
        raise FileNotFoundError(f"Replay dataset not found: {args.root}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Advantage file exists: {args.output}")

    device = resolve_device(args.device)
    policy, config, preprocessor, _ = _load_policy(args.value_policy, device)
    if config.type != "hepha_act_awr" or config.awr_stage != "value":
        raise ValueError("Advantages require a fitted hepha_act_awr value-stage checkpoint")

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    returns = discounted_return_map(dataset, discount=config.awr_discount)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    indices_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    preprocessor.reset()
    for batch in tqdm(loader, desc="Frozen critic inference"):
        absolute_indices = batch["index"].detach().cpu().numpy().reshape(-1)
        for camera_key in dataset.meta.camera_keys:
            if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                batch[camera_key] = batch[camera_key].float().div(255.0)
        with torch.inference_mode():
            processed = preprocessor(batch)
            values = policy.predict_value(processed)
        indices_parts.append(absolute_indices.astype(np.int64, copy=False))
        value_parts.append(values.detach().cpu().numpy().astype(np.float32, copy=False))

    indices = np.concatenate(indices_parts)
    values = np.concatenate(value_parts)
    return_values = np.asarray([returns[int(index)] for index in indices], dtype=np.float32)
    advantages = return_values - values
    advantage_mean = float(advantages.mean())
    advantage_std = float(advantages.std())
    if config.awr_normalize_advantage:
        normalized = (advantages - advantage_mean) / max(advantage_std, 1e-6)
    else:
        normalized = advantages.copy()
    maximum_log_weight = np.log(config.awr_max_weight)
    weights = np.exp(
        np.minimum(normalized / config.awr_beta, maximum_log_weight)
    ).astype(np.float32)

    episode_by_index = dict(
        zip(
            np.asarray(dataset.hf_dataset["index"], dtype=np.int64).tolist(),
            np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64).tolist(),
            strict=True,
        )
    )
    table = pa.table(
        {
            "index": indices,
            "episode_index": np.asarray(
                [episode_by_index[int(index)] for index in indices], dtype=np.int64
            ),
            "return": return_values,
            "value": values,
            "advantage": advantages.astype(np.float32),
            "normalized_advantage": normalized.astype(np.float32),
            "weight": weights,
        }
    )
    metadata = {
        **(table.schema.metadata or {}),
        b"hepha_awr": json.dumps(
            {
                "repo_id": args.repo_id,
                "value_policy": str(args.value_policy),
                "discount": config.awr_discount,
                "beta": config.awr_beta,
                "maximum_weight": config.awr_max_weight,
                "normalized_advantage": config.awr_normalize_advantage,
            },
            sort_keys=True,
        ).encode(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), args.output)
    print(f"Fixed advantages written to {args.output}")
    print(f"Frames: {len(indices)}")
    print(f"Advantage mean/std: {advantage_mean:.6f}/{advantage_std:.6f}")
    print(
        f"Weight mean/min/max: {weights.mean():.6f}/"
        f"{weights.min():.6f}/{weights.max():.6f}"
    )
    return args.output


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
