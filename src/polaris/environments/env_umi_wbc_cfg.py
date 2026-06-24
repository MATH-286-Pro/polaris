import torch
import numpy as np
import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseCfg, NoiseModelCfg
from isaaclab.assets import Articulation

from . import mdp as mdp

from .env_basic_cfg import SceneCfg, BasicEnvCfg
from .. import tool_linalg
from .sensor.camera import GoPro_2_7K
from .mdp.events import set_material_mirror, set_material_friction, set_passive_finger_joint_limits, reset_vx300s_arm_by_ee_pose
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


def _viewer_recording_camera_cfg(eye, lookat) -> CameraCfg:
    eye = np.asarray(eye, dtype=float)
    lookat = np.asarray(lookat, dtype=float)

    # Use a longer lens from farther away to reduce wide-angle perspective distortion
    # while keeping a similar recording composition.
    eye = lookat + 1.45 * (eye - lookat)

    forward = lookat - eye
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # IsaacLab's "opengl" camera convention looks along local -Z with +Y up.
    rot = np.column_stack([right, up, -forward])
    quat = tool_linalg._rot_to_quat(rot)

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/scene/realtime_default_cam",
        height=1440,
        width=2560,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.6,
            horizontal_aperture=2.5452,
            vertical_aperture=1.4721,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in eye),
            rot=tuple(float(v) for v in quat),
            convention="opengl",
        ),
    )


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


    reset_eef_pos = EventTerm(
        func=reset_vx300s_arm_by_ee_pose,
        mode="reset",
        params={
            "x_range": (0.2, 0.2),
            "z_range": (-0.3, -0.3),
            "yaw_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": UNITREE_A2_VX300S_ARM_JOINT_NAMES,
        },
    )


    reset_arm_elbow = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_angle"]),
            "position_range": (np.deg2rad(-60.0), np.deg2rad(-60.0)),  # degree
            "velocity_range": (0.0, 0.0),
        },
    )

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
#00ff00 临时
from .mdp import tools
from polaris.robot.robot_cfg import UNITREE_A2_VX300S_OFFSET_EEF_TF_E

def ee_target_pos_b(env: ManagerBasedRLEnv):
    # USE ZEROS
    target = torch.zeros(env.num_envs, WBC_TARGET_TRAJECTORY_LENGTH * WBC_EE_KEYPOINT_DIM, device=env.device)
    return target

def _visualize_ee_target_pos_b_debug(env: ManagerBasedRLEnv, eef_tf_w: torch.Tensor) -> None:
    visualizer = getattr(env, "_polaris_wbc_debug_target_visualizer", None)
    if visualizer is None:
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/WBC/ee_target_pos_b_debug"
        marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        visualizer = VisualizationMarkers(marker_cfg)
        setattr(env, "_polaris_wbc_debug_target_visualizer", visualizer)

    visualizer.set_visibility(True)
    target_pos_w = eef_tf_w[:, :3, 3]
    target_quat_w = tool_linalg.quat_from_matrix(eef_tf_w[:, :3, :3])
    visualizer.visualize(target_pos_w, target_quat_w)

def ee_target_pos_b_debug(env: ManagerBasedRLEnv):
    robot = env.scene["robot"]
    base_link_index = robot.find_bodies(EE_REF_LINK_NAME)[0][0]
    base_tf_w = tool_linalg.pose_2_tf(robot.data.body_pose_w[:, base_link_index])

    env_step = env.episode_length_buf[0]
    env_real_time = env_step * env.step_dt

    pos = torch.tensor([0.08, 0.0, 0.3], dtype=base_tf_w.dtype, device=env.device)
    rot = tool_linalg.euler_rad_to_rot(np.array([0.0, np.deg2rad(45), 0.0]))
    rot = torch.tensor(rot, dtype=base_tf_w.dtype, device=env.device)
    eef_tf_w = torch.eye(4, dtype=base_tf_w.dtype, device=env.device).unsqueeze(0).repeat(env.num_envs, 1, 1)
    eef_tf_w[:, :3, 3] = pos
    eef_tf_w[:, :3,:3] = rot

    if env_real_time <= 0.1:
        _visualize_ee_target_pos_b_debug(env, eef_tf_w)
    else:
        visualizer = getattr(env, "_polaris_wbc_debug_target_visualizer", None)
        if visualizer is not None:
            visualizer.set_visibility(False)

    world_tf_b = torch.linalg.inv(base_tf_w)
    eef_tf_b = world_tf_b @ eef_tf_w
    eef_tf_b = eef_tf_b @ torch.tensor(UNITREE_A2_VX300S_OFFSET_EEF_TF_E, device=env.device, dtype=torch.float32)

    target_pos_b = eef_tf_b[:, :3, 3]
    axis_x_b = eef_tf_b[:, :3, 0]
    axis_z_b = eef_tf_b[:, :3, 2]

    eef_3kp_b = torch.cat(
        [
            target_pos_b,
            target_pos_b + CUBE_LENGTH * axis_x_b,
            target_pos_b + CUBE_LENGTH * axis_z_b,
        ],
        dim=1,
    )

    eef_traj_3kp_b = eef_3kp_b.repeat(1, WBC_TARGET_TRAJECTORY_LENGTH)

    return eef_traj_3kp_b

def temp_eef_body_key_points(
    env: ManagerBasedRLEnv,
    link_name: str,
    ref_link_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_length: float = 0.3,
    ):

    robot = env.scene[asset_cfg.name]

    # 获取 EEF 位置 _w
    body_id = robot.find_bodies(link_name)[0][0]
    pos_quat_w = robot.data.body_pose_w[:, body_id]
    tf_w    = tool_linalg.pose_2_tf(pos_quat_w)

    # 获取 EEF 位置 _b
    ref_body_id = robot.find_bodies(ref_link_name)[0][0]
    ref_pos_quat_w = robot.data.body_pose_w[:, ref_body_id]
    ref_tf_w    = tool_linalg.pose_2_tf(ref_pos_quat_w)
    eef_tf_b = tool_linalg.tf_reference(tf_w, ref_tf_w)

    # 转为 EEF OFFSET
    eef_train_tf_eef = torch.tensor(UNITREE_A2_VX300S_OFFSET_EEF_TF_E, device=env.device, dtype=torch.float32)
    eef_train_tf_b = eef_tf_b @ eef_train_tf_eef[None, :]

    # 转为 3kp 表示
    current_link_pos_b, current_link_rot_b = tools.tf_2_pos_rot(eef_train_tf_b)

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
            func=ee_target_pos_b_debug,  # DEBUG #00ff00
        )

        # ee_current_pos_b = ObsTerm(
        #     func=mdp.body_keypoints,
        #     params={
        #         "current_link_name": EE_LINK_NAME,
        #         "reference_link_name": EE_REF_LINK_NAME,
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "cube_length": CUBE_LENGTH,
        #     },
        #     clip=DEFAULT_OBS_CLIP,
        # )

        ee_current_pos_b = ObsTerm(
            func=temp_eef_body_key_points,
            params={
                "link_name": EE_LINK_NAME,
                "ref_link_name": EE_REF_LINK_NAME,
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
            pos=(0.003, 0.0, 0.0),
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

        self.viewer.eye = (0.8, 0.5, 0.8)
        self.viewer.lookat = (0.0, -0.1, 0.3)

        self.sim.dt = 1 / 200
        self.decimation = 4
        self.sim.render_interval = 4

        # self.sim.gravity = (0.0, 0.0, 0.0)  # DEBUG #00ff00

        self.rerender_on_reset = True

    def dynamic_setup(self, *args):
        self.scene.dynamic_setup(*args)
        self.scene.external_cam = _viewer_recording_camera_cfg(
            self.viewer.eye,
            self.viewer.lookat,
        )
