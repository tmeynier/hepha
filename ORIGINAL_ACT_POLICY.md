# Original ACT Policy

## Implementation

- Standard LeRobot ACT policy.
- ResNet-18 visual backbone.
- ACT transformer with CVAE training.
- Action chunk: 100 steps.
- Closed-loop evaluation used one action per frame.
- Temporal-ensemble coefficient: `0.01`.

## Input

- Head-camera RGB image: `3 × 256 × 256`.
- Robot joint state: 15 joint positions.
- Requested drawer: 9-value one-hot vector.

## Output

- Absolute joint-position targets.
- Shape: `100 × 15` per predicted action chunk.

## Dataset and model

- Training dataset: `tmeynier/hepha_mujoco_ik` on Hugging Face.
- RunPod dataset path: `/workspace/datasets/hepha_mujoco_ik`.
- Local dataset is not currently stored in this repository.
- Model: `tmeynier/hepha_act_200` on Hugging Face.
- Local model: `models/hepha_act_200`.

## Closed-loop results

- Evaluation: 100 episodes of 100 seconds.
- Seeds: `10000–10099`.
- Domain randomization: disabled.
- Drawer opened at least 40 mm: **100%**.
- Cube stably grasped: **32%**.
- Cube entered the selected drawer: **11%**.
- Drawer closed to at most 5 mm after insertion: **11%**.
- Final cube-inside-and-drawer-closed success: **2%**.
- Successful final-state seeds: `10024`, `10029`.
- Task reached temporarily: **11%**.
- Task retained until the end: **2%**.
