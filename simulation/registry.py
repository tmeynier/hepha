"""Lazy registry for built-in and externally installed simulation backends."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

from .base import SimulationBackend, SimulationConfig

ENTRY_POINT_GROUP = "hepha.simulation_backends"
BUILTIN_BACKENDS = {
    "mujoco": "simulation.backends.mujoco.backend:MujocoBackend",
}


def _load_path(path: str) -> type[SimulationBackend]:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError(f"Backend path must use module:class syntax: {path}")
    backend_class = getattr(import_module(module_name), attribute)
    if not isinstance(backend_class, type) or not issubclass(
        backend_class, SimulationBackend
    ):
        raise TypeError(f"Registered backend is not a SimulationBackend: {path}")
    return backend_class


def _external_backends() -> dict[str, Any]:
    return {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)}


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(set(BUILTIN_BACKENDS) | set(_external_backends())))


def get_backend_class(name: str) -> type[SimulationBackend]:
    if name in BUILTIN_BACKENDS:
        return _load_path(BUILTIN_BACKENDS[name])
    external = _external_backends().get(name)
    if external is None:
        choices = ", ".join(available_backends()) or "none installed"
        raise ValueError(f"Unknown simulation backend {name!r}; available: {choices}")
    backend_class = external.load()
    if not isinstance(backend_class, type) or not issubclass(
        backend_class, SimulationBackend
    ):
        raise TypeError(f"Entry point {name!r} is not a SimulationBackend")
    return backend_class


def create_backend(name: str, config: SimulationConfig) -> SimulationBackend:
    return get_backend_class(name)(config)
