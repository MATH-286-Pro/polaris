import torch
import numpy as np
import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseCfg, NoiseModelCfg
from isaaclab.assets import Articulation

from . import mdp as mdp

from .env_basic_cfg import SceneCfg, BasicEnvCfg
from .. import tool_linalg
from .sensor.camera import GoPro_2_7K
from .mdp.events import set_material_mirror, set_material_friction, set_passive_finger_joint_limits
from .mdp.observations import body_tf_b_priv, body_tf_w_priv, gripper_pos, arm_joint_pos
from .mdp.actions import GripperWidthJointPositionActionCfg

from dataclasses import MISSING

from polaris.robot.robot_cfg import (
    RobotFrameCfg,
    UNITREE_A2_VX300S_CFG,
    UNITREE_A2_VX300S_ROOT_PRIM_PATH,
    UNITREE_A2_VX300S_CAMERA_PRIM_PATH,
    UNITREE_A2_VX300S_ARM_JOINT_NAMES,
    UNITREE_A2_VX300S_FINGER_JOINT_NAMES,
    UNITREE_A2_VX300S_WBC_JOINT_NAMES,
    UNITREE_A2_VX300S_FINGER_BODY_NAMES,
    UNITREE_A2_VX300S_PASSIVE_FINGER_JOINTS,
    UNITREE_A2_VX300S_FRAME_CFG,
    EE_LINK_NAME,
    EE_REF_LINK_NAME,
)

DEFAULT_OBS_CLIP = (-100.0, 100.0)  # applied before scaling

CUBE_LENGTH      = 0.3

WBC_TRAJ_OFFSET = [0.0, +0.02, +0.04, +0.06, +1.0]

WBC_TARGET_TRAJECTORY_LENGTH = 5
WBC_EE_KEYPOINT_DIM = 9
WBC_EE_TARGET_DIM = len(WBC_TRAJ_OFFSET) * WBC_EE_KEYPOINT_DIM

# ======================== Action ========================#
@configclass
class ActionCfg:
    joint_pos = mdp.JointPositionActionClippedCfg(
        asset_name="robot",
        joint_names=UNITREE_A2_VX300S_WBC_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        pre_clip=(-20, +20),
    )

    finger_joint = GripperWidthJointPositionActionCfg(
        asset_name="robot",
        joint_names=UNITREE_A2_VX300S_FINGER_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=True,
        width_to_joint_scale={
            "left_finger_joint": 0.5,
            "right_finger_joint": 0.5,
        },
    )

# ======================== Event ======================== #
@configclass
class EventCfg:

    set_material_mirror = EventTerm(
        func=set_material_mirror,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*mirror"]),
        }, 
    )

    set_material_friction_fingers = EventTerm(
        func=set_material_friction,
        mode="startup",
        params={
            "static_friction": 1.5,
            "dynamic_friction": 1.5,
            "friction_combine_mode": "average",
            "asset_cfg": SceneEntityCfg("robot", body_names=UNITREE_A2_VX300S_FINGER_BODY_NAMES),
        },
    )

    set_passive_finger_joint_limits = EventTerm(
        func=set_passive_finger_joint_limits,
        mode="startup",
        params={
            "lower": 0.0,
            "upper": 0.02,
            "joint_names": UNITREE_A2_VX300S_PASSIVE_FINGER_JOINTS,
        },
    )

    initial_joint_pos = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, -0.5), 
                    "y": (-0.0, 0.0), 
                    "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "z": (-0.0, 0.0),
                    "roll": (-0.0, 0.0),
                    "pitch": (-0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
            },
    )

    reset_all_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


    # reset_arm_elbow = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_angle"]),
    #         "position_range": (np.deg2rad(30.0), np.deg2rad(30.0)),  # degree
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    open_claw = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_A2_VX300S_FINGER_JOINT_NAMES),
            "position_range": (0.04, 0.04),
            "velocity_range": (0.0, 0.0),
        },    
    )

# ======================== Observation ========================#
def ee_target_pos_b(env: ManagerBasedRLEnv):
    # USE ZEROS
    target = torch.zeros(env.num_envs, WBC_TARGET_TRAJECTORY_LENGTH * WBC_EE_KEYPOINT_DIM, device=env.device)
    return target


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        # ======================== High Level Policy: UMI ========================= #

        eef_tf_b_priv = ObsTerm(
            func=body_tf_b_priv,
            params={
                "link_name": EE_LINK_NAME,
                "ref_link_name": EE_REF_LINK_NAME,
            }
        )

        eef_tf_w_priv = ObsTerm(
            func=body_tf_w_priv,
            params={
                "link_name": EE_LINK_NAME,
            }
        )

        base_tf_w_priv = ObsTerm(
            func=body_tf_w_priv,
            params={
                "link_name": "base_link",
            }
        )

        arm_joint_pos = ObsTerm(
            func=arm_joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_A2_VX300S_ARM_JOINT_NAMES)
            },
        )

        gripper_pos = ObsTerm(
            func=gripper_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_A2_VX300S_FINGER_JOINT_NAMES)
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class WbcCfg(ObsGroup):
        """A2-vx300s low-level neural WBC observations."""

        ee_target_pos_b = ObsTerm(
            func=ee_target_pos_b,
        )

        ee_current_pos_b = ObsTerm(
            func=mdp.body_keypoints,
            params={
                "current_link_name": EE_LINK_NAME,
                "reference_link_name": EE_REF_LINK_NAME,
                "asset_cfg": SceneEntityCfg("robot"),
                "cube_length": CUBE_LENGTH,
            },
            clip=DEFAULT_OBS_CLIP,
        )

        base_lin_vel_b = ObsTerm(
            func=mdp.body_lin_vel_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=DEFAULT_OBS_CLIP,
        )

        base_ang_vel_b = ObsTerm(
            func=mdp.body_ang_vel_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=DEFAULT_OBS_CLIP,
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, 
            noise=Unoise(n_min=-0.05, n_max=0.05), 
            clip=DEFAULT_OBS_CLIP)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_A2_VX300S_WBC_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=DEFAULT_OBS_CLIP)

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_A2_VX300S_WBC_JOINT_NAMES)},
            noise=Unoise(n_min=-0.15, n_max=0.15),
            clip=DEFAULT_OBS_CLIP)

        last_actions = ObsTerm(
            func=mdp.last_action_clipped, 
            params={"action_name": "joint_pos"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    wbc: WbcCfg = WbcCfg()


# ======================== EnvCfg ========================#
@configclass
class UMIWBCEnvCfg(BasicEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    scene.robot = UNITREE_A2_VX300S_CFG.replace(prim_path=UNITREE_A2_VX300S_ROOT_PRIM_PATH)
    scene.wrist_cam = TiledCameraCfg(
        prim_path=UNITREE_A2_VX300S_CAMERA_PRIM_PATH,
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
            fisheye_polynomial_a=0.0,
            fisheye_polynomial_b=1 / GoPro_2_7K.fx,
            fisheye_polynomial_c=0.0,
        ),

        # Render output resolution.
        # This can be lower than the nominal calibration resolution.
        width=GoPro_2_7K.image_width,
        height=GoPro_2_7K.image_height,
    )


    observations = ObservationCfg()
    actions = ActionCfg()
    events = EventCfg()


    def __post_init__(self):
        self.scene.make_ee_frame_cfg(UNITREE_A2_VX300S_FRAME_CFG)
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

        self.episode_length_s = 30

        self.viewer.eye = (0.8, -0.8, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.2)

        self.sim.dt = 1 / 200
        self.decimation = 4
        self.sim.render_interval = 4

        # self.sim.gravity = (0.0, 0.0, 0.0)  # DEBUG #00ff00

        self.rerender_on_reset = True

    def dynamic_setup(self, *args):
        self.scene.dynamic_setup(*args)
