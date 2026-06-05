from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_body_names, record_dtype, record_shape
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import tools
from ... import tool_linalg

# ================================= User Defined ================================= #
def body_tf_b_priv(
    env: ManagerBasedRLEnv,
    link_name: str,
    ref_link_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot = env.scene[asset_cfg.name]
    body_id = robot.find_bodies(link_name)[0][0]
    pos_quat_w = robot.data.body_pose_w[:, body_id]
    tf_w    = tool_linalg.pose_2_tf(pos_quat_w)

    ref_body_id = robot.find_bodies(ref_link_name)[0][0]
    ref_pos_quat_w = robot.data.body_pose_w[:, ref_body_id]
    ref_tf_w    = tool_linalg.pose_2_tf(ref_pos_quat_w)

    tf_b = tool_linalg.tf_reference(tf_w, ref_tf_w)
    return tf_b


def body_tf_w_priv(
    env: ManagerBasedRLEnv,
    link_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot = env.scene[asset_cfg.name]
    body_id = robot.find_bodies(link_name)[0][0]
    pos_quat_w = robot.data.body_pose_w[:, body_id]
    tf_w    = tool_linalg.pose_2_tf(pos_quat_w)

    return tf_w

def arm_joint_pos(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    #TODO 这个是相对位置吗?
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    gripper_width = torch.sum(joint_pos, dim=1, keepdim=True)

    return gripper_width

# ================================= User Defined ================================= #



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv




def last_action_clipped(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """The last input clipped action to the environment.

    The name of the action term for which the action is required.
    """

    return env.action_manager.get_term(action_name).clipped_actions


@generic_io_descriptor(
    units="m/s",
    axes=["X", "Y", "Z"],
    observation_type="BodyState",
    on_inspect=[record_shape, record_dtype, record_body_names],
)
def body_lin_vel_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Linear velocity of specified bodies in their respective body frames.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset. Defaults to SceneEntityCfg("robot").
            Use body_ids or body_names to specify which bodies to observe.

    Returns:
        Linear velocity of shape (num_instances, num_bodies, 3) in each body's local frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get linear velocities in world frame for specified bodies
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]

    # Get orientations for specified bodies
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]

    # Transform velocities to body frame
    body_lin_vel_b = math_utils.quat_apply_inverse(body_quat_w, body_lin_vel_w)

    # Drop the link dimension if only one body is requested
    if len(asset_cfg.body_ids) == 1:
        body_lin_vel_b = body_lin_vel_b.squeeze(1)

    return body_lin_vel_b


@generic_io_descriptor(
    units="rad/s",
    axes=["X", "Y", "Z"],
    observation_type="BodyState",
    on_inspect=[record_shape, record_dtype, record_body_names],
)
def body_ang_vel_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocity of specified bodies in their respective body frames.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset. Defaults to SceneEntityCfg("robot").
            Use body_ids or body_names to specify which bodies to observe.

    Returns:
        Angular velocity of shape (num_instances, num_bodies, 3) in each body's local frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get angular velocities in world frame for specified bodies
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]

    # Get orientations for specified bodies
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]

    # Transform velocities to body frame
    body_ang_vel_b = math_utils.quat_apply_inverse(body_quat_w, body_ang_vel_w)

    # Drop the link dimension if only one body is requested
    if len(asset_cfg.body_ids) == 1:
        body_ang_vel_b = body_ang_vel_b.squeeze(1)

    return body_ang_vel_b


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_vel_last(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame at t-1.

    Stores a per-asset previous-step buffer on the env as
    `_prev_base_lin_vel_b_{asset_name}`. On first call, returns zeros and
    initializes the buffer. On subsequent calls returns the stored value and
    updates the buffer with the current value (cheap clone).

    Attention: this function must be called exactly once per env step to ensure
    correct behavior.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # per-asset attribute name to support multiple assets
    attr_name = f"_prev_base_lin_vel_b_{asset_cfg.name}"

    # initialize if missing (zeros)
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros_like(asset.data.root_lin_vel_b))

    prev = getattr(env, attr_name)

    # update stored previous value for the next step (cheap clone)
    setattr(env, attr_name, asset.data.root_lin_vel_b.clone())

    return prev


@generic_io_descriptor(
    units="m/s",
    axes=["X", "Y", "Z"],
    observation_type="BodyState",
    on_inspect=[record_shape, record_dtype, record_body_names],
)
def body_lin_vel_last(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Linear velocity of specified body in the body's frame at t-1.

    Stores a per-asset-body previous-step buffer on the env as
    `_prev_body_lin_vel_b_{asset_name}_{body_ids}`. On first call, returns zeros and
    initializes the buffer. On subsequent calls returns the stored value and
    updates the buffer with the current value (cheap clone).

    Attention: this function must be called exactly once per env step to ensure
    correct behavior.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # per-asset-body attribute name to support multiple assets and body selections
    body_ids_str = "_".join(map(str, asset_cfg.body_ids))
    attr_name = f"_prev_body_lin_vel_b_{asset_cfg.name}_{body_ids_str}"

    # Compute current body-frame linear velocity
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    current_body_lin_vel_b = math_utils.quat_apply_inverse(body_quat_w, body_lin_vel_w)

    # Drop the link dimension if only one body is requested
    if len(asset_cfg.body_ids) == 1:
        current_body_lin_vel_b = current_body_lin_vel_b.squeeze(1)

    # initialize if missing (zeros)
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros_like(current_body_lin_vel_b))

    prev = getattr(env, attr_name)

    # update stored previous value for the next step (cheap clone)
    setattr(env, attr_name, current_body_lin_vel_b.clone())

    return prev


@generic_io_descriptor(
    units="rad/s",
    axes=["X", "Y", "Z"],
    observation_type="BodyState",
    on_inspect=[record_shape, record_dtype, record_body_names],
)
def body_ang_vel_last(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocity of specified body in the body's frame at t-1.

    Stores a per-asset-body previous-step buffer on the env as
    `_prev_body_ang_vel_b_{asset_name}_{body_ids}`. On first call, returns zeros and
    initializes the buffer. On subsequent calls returns the stored value and
    updates the buffer with the current value (cheap clone).

    Attention: this function must be called exactly once per env step to ensure
    correct behavior.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # per-asset-body attribute name to support multiple assets and body selections
    body_ids_str = "_".join(map(str, asset_cfg.body_ids))
    attr_name = f"_prev_body_ang_vel_b_{asset_cfg.name}_{body_ids_str}"

    # Compute current body-frame angular velocity
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    current_body_ang_vel_b = math_utils.quat_apply_inverse(body_quat_w, body_ang_vel_w)

    # Drop the link dimension if only one body is requested
    if len(asset_cfg.body_ids) == 1:
        current_body_ang_vel_b = current_body_ang_vel_b.squeeze(1)

    # initialize if missing (zeros)
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros_like(current_body_ang_vel_b))

    prev = getattr(env, attr_name)

    # update stored previous value for the next step (cheap clone)
    setattr(env, attr_name, current_body_ang_vel_b.clone())

    return prev


@generic_io_descriptor(
    units="m",
    axes=["X", "Y", "Z"],
    observation_type="BodyState",
    on_inspect=[record_shape, record_dtype, record_body_names],
)
def body_keypoints(
    env: ManagerBasedEnv,
    current_link_name: str,
    reference_link_name: str,
    cube_length: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of specified bodies expressed in a base body's frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset. Defaults to SceneEntityCfg("robot").
            Use body_ids or body_names to specify which target bodies to observe.
    Returns:
        Position of shape (num_instances, num_bodies, 3) in the base body frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    current_link_index   = asset.find_bodies(current_link_name)[0][0]
    reference_link_index = asset.find_bodies(reference_link_name)[0][0]


    # transform
    current_link_pose_w = asset.data.body_pose_w[:, current_link_index]
    current_link_tf_w   = tools.pose_2_tf(current_link_pose_w)

    reference_link_pose_w = asset.data.body_pose_w[:, reference_link_index]
    reference_link_tf_w   = tools.pose_2_tf(reference_link_pose_w)

    current_link_tf_b = torch.linalg.inv(reference_link_tf_w) @ current_link_tf_w
    current_link_pos_b, current_link_rot_b = tools.tf_2_pos_rot(current_link_tf_b)

    # 3 key-points representation
    axis_x_b = current_link_rot_b[:, :, 0]
    axis_y_b = current_link_rot_b[:, :, 1]
    axis_z_b = current_link_rot_b[:, :, 2]

    current_link_pos_c_b = current_link_pos_b
    current_link_pos_x_b = current_link_pos_b + cube_length * axis_x_b
    current_link_pos_z_b = current_link_pos_b + cube_length * axis_z_b

    current_link_3key_points_b = torch.cat(
        [current_link_pos_c_b, 
         current_link_pos_x_b, 
         current_link_pos_z_b],
         dim=1
    )

    return current_link_3key_points_b



