"""Canonical Hepha servo-to-joint definitions shared by hardware utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisDefinition:
    label: str
    mujoco_actuator: str
    q_min: float
    q_max: float
    encoder_direction: int

    @property
    def joint_name(self) -> str:
        """Hardware-neutral semantic name used between leaders and followers."""
        return self.mujoco_actuator


AXES = {
    1: AxisDefinition("shoulder right", "shoulder_r", -math.pi / 2, math.pi / 2, 1),
    2: AxisDefinition("shoulder left", "shoulder_l", -math.pi / 2, math.pi / 2, 1),
    3: AxisDefinition("forearm right", "forearm_r", -math.pi / 4, 3 * math.pi / 4, -1),
    4: AxisDefinition("forearm left", "forearm_l", -math.pi / 4, 3 * math.pi / 4, 1),
    5: AxisDefinition("arm right", "arm_r", -3 * math.pi / 4, math.pi / 4, -1),
    6: AxisDefinition("arm left", "arm_l", -3 * math.pi / 4, math.pi / 4, 1),
    7: AxisDefinition("wrist right", "wrist_r", -math.pi / 2, math.pi / 2, -1),
    8: AxisDefinition("wrist left", "wrist_l", -math.pi / 2, math.pi / 2, -1),
    9: AxisDefinition("hand right", "hand_r", -math.pi / 2, math.pi / 2, -1),
    10: AxisDefinition("hand left", "hand_l", -math.pi / 2, math.pi / 2, 1),
    11: AxisDefinition("finger right", "finger_r", 0.0, math.pi / 2, -1),
    12: AxisDefinition("finger left", "finger_l", 0.0, math.pi / 2, 1),
}

SERVO_LABELS = {servo_id: axis.label for servo_id, axis in AXES.items()}
AXES_BY_JOINT = {axis.joint_name: axis for axis in AXES.values()}
