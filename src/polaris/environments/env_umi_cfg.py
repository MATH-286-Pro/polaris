import torch
from pathlib import Path
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math
import isaaclab.envs.mdp as mdp
import numpy as np
from typing import Sequence

from polaris.robot.robot_cfg import (
    UMI_EE_LINK_NAME,
    UMI_REF_EE_LINK_NAME,
    RobotFrameCfg,
    UMI_GRIPPER,
    UMI_GRIPPER_CAMERA_PRIM_PATH,
    UMI_GRIPPER_FRAME_CFG,
    UMI_GRIPPER_FINGER_JOINTS,
    UMI_GRIPPER_ARM_JOINTS,
)

from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg, Camera, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import dataclasses

from .env_basic_cfg import EventCfg as BasicEventCfg
from .env_basic_cfg import SceneCfg, BasicEnvCfg
from .mdp.observations import body_tf_b_priv, body_tf_w_priv, arm_joint_pos, gripper_pos
from .mdp.actions import GripperWidthJointPositionActionCfg
from .. import tool_linalg
from .sensor.camera import GoPro_2_7K


# ======================== Action ========================#
@configclass
class ActionCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = GripperWidthJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_finger_joint",
            "right_finger_joint",
        ],
        preserve_order=True,
        use_default_offset=False,
        width_to_joint_scale={
            "left_finger_joint": 0.5,
            "right_finger_joint": 0.5,
        },
    )



# ======================== Observation ========================#
@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(
            func=arm_joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UMI_GRIPPER_ARM_JOINTS)
            },
        )

        gripper_pos = ObsTerm(
            func=gripper_pos,            
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UMI_GRIPPER_FINGER_JOINTS)
            },
        )

        eef_tf_b_priv = ObsTerm(
            func=body_tf_b_priv,
            params={
                "link_name": UMI_EE_LINK_NAME,
                "ref_link_name": UMI_REF_EE_LINK_NAME,
                "asset_cfg": SceneEntityCfg("robot"),
            }
        )

        eef_tf_w_priv = ObsTerm(
            func=body_tf_w_priv,
            params={
                "link_name": UMI_EE_LINK_NAME,
                "asset_cfg": SceneEntityCfg("robot"),
            }
        )

        base_tf_w_priv = ObsTerm(
            func=body_tf_w_priv,
            params={
                "link_name": UMI_REF_EE_LINK_NAME,
                "asset_cfg": SceneEntityCfg("robot"),
            }
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# ======================== Event ========================#
@configclass
class EventCfg(BasicEventCfg):

    # In BasicEventCfg there are
    # reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    set_joint_x = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint_x"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }, 
    )

    set_joint_z = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint_z"]),
            "position_range": (+0.4, +0.4),
            "velocity_range": (0.0, 0.0),
        }, 
    )

    set_joint_pitch = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint_ry"]),
            "position_range": (np.deg2rad(45), np.deg2rad(45)),
            "velocity_range": (0.0, 0.0),
        }, 
    )

    open_claw = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_finger_joint", "right_finger_joint"]),
            "position_range": (0.0425, 0.0425),
            "velocity_range": (0.0, 0.0),
        },    
    )

# ======================== EnvCfg ========================#
@configclass
class UMIEnvCfg(BasicEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    scene.robot = UMI_GRIPPER
    scene.wrist_cam = TiledCameraCfg(
        prim_path=UMI_GRIPPER_CAMERA_PRIM_PATH,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyePolynomial",

            fisheye_nominal_width=GoPro_2_7K.image_width,
            fisheye_nominal_height=GoPro_2_7K.image_height,
            fisheye_optical_centre_x=GoPro_2_7K.cx,
            fisheye_optical_centre_y=GoPro_2_7K.cy,
            fisheye_max_fov=GoPro_2_7K.fov,

            # Fisheye distortion coefficients.
            fisheye_polynomial_a=GoPro_2_7K.D[0],
            fisheye_polynomial_b=GoPro_2_7K.D[1],
            fisheye_polynomial_c=GoPro_2_7K.D[2],
            fisheye_polynomial_d=GoPro_2_7K.D[3],
            fisheye_polynomial_e=GoPro_2_7K.D[4],
            fisheye_polynomial_f=GoPro_2_7K.D[5],
        ),

        # Render output resolution.
        # This can be lower than the nominal calibration resolution.
        width=GoPro_2_7K.image_width,
        height=GoPro_2_7K.image_height,
    )


    observations = ObservationCfg()
    actions = ActionCfg()


    def __post_init__(self):
        self.scene.make_ee_frame_cfg(UMI_GRIPPER_FRAME_CFG)

        self.episode_length_s = 30

        self.viewer.eye = (0.8, -0.8, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.2)

        self.decimation = 4 * 2
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 4 * 2

        self.rerender_on_reset = True

    def dynamic_setup(self, *args):
        self.scene.dynamic_setup(*args)
