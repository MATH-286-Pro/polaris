import torch

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ... import tool_linalg


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
