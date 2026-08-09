"""Lazy registry for demonstration sources used during recording."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Any, Protocol

from simulation import SimulationBackend

ENTRY_POINT_GROUP = "hepha.demonstration_controllers"
BUILTIN_CONTROLLERS = {
    ("mujoco", "ik"): (
        "simulation.backends.mujoco.ik:MujocoIKController"
    ),
}


class DemonstrationController(Protocol):
    def reset(self, *, episode_seed: int) -> None: ...

    def action(self, progress: float) -> Any: ...


def _load_path(path: str) -> Any:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError(f"Controller path must use module:class syntax: {path}")
    return getattr(import_module(module_name), attribute)


def _external_controllers() -> dict[str, Any]:
    return {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)}


def available_controllers(backend: str) -> tuple[str, ...]:
    builtins = {name for backend_name, name in BUILTIN_CONTROLLERS if backend_name == backend}
    prefix = f"{backend}."
    external = {
        name.removeprefix(prefix)
        for name in _external_controllers()
        if name.startswith(prefix)
    }
    return tuple(sorted(builtins | external))


def create_controller(
    name: str,
    *,
    backend: SimulationBackend,
    seed: int,
) -> DemonstrationController:
    key = (backend.name, name)
    if key in BUILTIN_CONTROLLERS:
        controller_class = _load_path(BUILTIN_CONTROLLERS[key])
    else:
        external = _external_controllers().get(f"{backend.name}.{name}")
        if external is None:
            choices = ", ".join(available_controllers(backend.name)) or "none installed"
            raise ValueError(
                f"Unknown controller {name!r} for backend {backend.name!r}; "
                f"available: {choices}"
            )
        controller_class = external.load()
    return controller_class(backend, seed=seed)
