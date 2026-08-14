from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("hepha MuJoCo IK sweep")
    group.addoption(
        "--ik-seeds",
        default=None,
        metavar="SPEC",
        help=(
            "Run the headless IK episode sweep for comma-separated seeds or "
            "half-open ranges, for example 0:100 or 0,4,15:20."
        ),
    )
    group.addoption(
        "--ik-episode-seconds",
        type=float,
        default=90.0,
        metavar="SECONDS",
        help="Maximum simulated duration for each IK sweep episode (default: 90).",
    )


def _parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            seeds.append(int(item))
            continue
        values = [int(value) if value else None for value in item.split(":")]
        if len(values) not in (2, 3) or values[1] is None:
            raise ValueError(f"Invalid seed range: {item!r}")
        start = values[0] if values[0] is not None else 0
        step = values[2] if len(values) == 3 and values[2] is not None else 1
        if step == 0:
            raise ValueError(f"Seed range step cannot be zero: {item!r}")
        seeds.extend(range(start, values[1], step))
    if not seeds:
        raise ValueError("--ik-seeds must select at least one seed")
    return list(dict.fromkeys(seeds))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "ik_seed" not in metafunc.fixturenames:
        return
    spec = metafunc.config.getoption("--ik-seeds")
    if spec is None:
        metafunc.parametrize(
            "ik_seed",
            [
                pytest.param(
                    0,
                    marks=pytest.mark.skip(
                        reason="pass --ik-seeds to run the headless IK episode sweep"
                    ),
                )
            ],
        )
        return
    try:
        seeds = _parse_seed_spec(spec)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    metafunc.parametrize("ik_seed", seeds, ids=lambda seed: f"seed-{seed}")
