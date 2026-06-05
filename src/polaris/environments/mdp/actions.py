from __future__ import annotations
import torch

from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from dataclasses import MISSING

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

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


class JointPositionActionClipped(mdp.JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: JointPositionActionClippedCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointPositionActionClippedCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # disable default clipping (which is applied after scaling)
        if self.cfg.clip is not None:
            raise ValueError(f"Normal clipping is not supported, use pre_clip instead")

        self._clipped_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        if cfg.pre_clip is not None:
            self._min_clip = torch.tensor(cfg.pre_clip[0]).to(self.device)
            self._max_clip = torch.tensor(cfg.pre_clip[1]).to(self.device)

    """
    Properties.
    """

    @property
    def clipped_actions(self) -> torch.Tensor:
        return self._clipped_actions
    
    """
    Operations.
    """
        
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions

        # apply pre-clipping and store result (to be used as observation)
        if self.cfg.pre_clip is not None:
            self._clipped_actions = torch.clip(self._raw_actions, self._min_clip, self._max_clip) #.to(self.device)
        else:
            self._clipped_actions[:] = actions


        # apply the affine transformations
        self._processed_actions = self._clipped_actions * self._scale + self._offset
        
        # do NOT apply default clip clip actions
        # if self.cfg.clip is not None:
        #     self._processed_actions = torch.clamp(
        #         self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
        #     )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


def last_action_clipped(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """The last clipped input action for an action term."""
    return env.action_manager.get_term(action_name).clipped_actions


@configclass
class JointPositionActionClippedCfg(mdp.JointActionCfg):
    """Configuration for the joint position action term, with clipping applied first as in isaac gym

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = JointPositionActionClipped

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
    
    pre_clip: tuple[float, float] | None = None
    """The clipping range, applied first. Use this instead of the base clip """
