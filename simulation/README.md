# Simulation backends

`simulation.base.SimulationBackend` is the stable boundary between Hepha,
LeRobot, and a simulator. Built-in backends are registered lazily, so MuJoCo is
not imported when selecting another environment.

Third-party or future Isaac Sim backends can register a `SimulationBackend`
subclass with the `hepha.simulation_backends` entry-point group. Backend-specific
recording controllers use `hepha.demonstration_controllers` and the entry-point
name `<backend>.<controller>`.
