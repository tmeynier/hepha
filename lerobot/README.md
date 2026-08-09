# Hepha's LeRobot integration

This directory contains Hepha's LeRobot adapters and workflow entry points. The
official `lerobot` PyPI package remains the source of truth for datasets, policy
implementations, training, preprocessing, checkpoints, and rollout.

This directory deliberately has no `__init__.py`. That prevents it from
shadowing the official `lerobot` package when commands run from the project root.
Packaging maps these files to an internal `hepha_lerobot` import namespace, but
there is no corresponding extra directory in the project tree.

```text
datasets/                    Feature/frame construction using official LeRobot APIs
policies/                    Discovery of every policy registered by installed LeRobot
recording/                   Backend-neutral episode recording
training/                    Policy-neutral launcher for lerobot-train
evaluation/                  Backend-neutral launcher for lerobot-rollout
config_hepha_simulation.py   LeRobot robot configuration
hepha_simulation.py          Backend-neutral LeRobot Robot adapter
lerobot_robot_hepha.py       Required LeRobot third-party discovery shim
```
