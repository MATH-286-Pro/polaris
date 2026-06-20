import torch

from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from dataclasses import MISSING

UMI_GRIPPER_MAX_WIDTH = 0.085  # meter, distance between the two fingers
UMI_GRIPPER_MIN_WIDTH = 0

class GripperWidthJointPositionAction(JointPositionAction):
    """Map one UMI aperture-width command to multiple finger joints."""

    cfg: "GripperWidthJointPositionActionCfg"

    def __init__(self, cfg: "GripperWidthJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._width_to_joint_scale = torch.tensor(
            [cfg.width_to_joint_scale[name] for name in self._joint_names],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        width = torch.clamp(
            actions.to(dtype=torch.float32),
            min=self.cfg.min_width,
            max=self.cfg.max_width,
        )
        self._processed_actions = width * self._width_to_joint_scale


@configclass
class GripperWidthJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for one-dimensional UMI gripper-width control."""

    class_type = GripperWidthJointPositionAction
    width_to_joint_scale: dict[str, float] = MISSING
    min_width: float = UMI_GRIPPER_MIN_WIDTH
    max_width: float = UMI_GRIPPER_MAX_WIDTH