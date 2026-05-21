from pathlib import Path
from dataclasses import dataclass

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


ROBOT_PATH = Path(__file__).resolve().parents[3] / "robot_descriptions"


@dataclass(frozen=True)
class RobotFrameCfg:
    root_prim_path: str
    source_frame_prim_path: str
    end_effector_prim_path: str
    end_effector_name: str = "eef_link"


DROID_ROOT_PRIM_PATH = "{ENV_REGEX_NS}/robot"
DROID_BASE_FRAME_PRIM_PATH = f"{DROID_ROOT_PRIM_PATH}/panda_link0"
DROID_END_EFFECTOR_PRIM_PATH = f"{DROID_ROOT_PRIM_PATH}/Gripper/Robotiq_2F_85/base_link"
DROID_WRIST_CAMERA_PRIM_PATH = f"{DROID_END_EFFECTOR_PRIM_PATH}/wrist_cam"
DROID_FRAME_CFG = RobotFrameCfg(
    root_prim_path=DROID_ROOT_PRIM_PATH,
    source_frame_prim_path=DROID_BASE_FRAME_PRIM_PATH,
    end_effector_prim_path=DROID_END_EFFECTOR_PRIM_PATH,
)


NVIDIA_DROID = ArticulationCfg(
    prim_path=DROID_ROOT_PRIM_PATH,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_PATH}/nvidia_droid/noninstanceable.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * np.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * np.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * np.pi,
            "panda_joint7": 0,
            "finger_joint": 0.0,
            "right_outer.*": 0.0,
            "left_inner.*": 0.0,
            "right_inner.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=None,
            damping=None,
            effort_limit=200.0,
            velocity_limit=5.0,  # 2.175,
        ),
    },
)


UMI_EE_LINK_NAME = "eef_link"
UMI_REF_EE_LINK_NAME = "world"

# Offset from the commanded joint-origin frame to the UMI gripper target frame,
# expressed in the target/body frame. Keep the historical name for callers, but
# do not subtract it directly in world coordinates.
UMI_EE_TO_JOINT_ORIGIN_B = np.array([0.14, 0.0, 0.0], dtype=float)

UMI_GRIPPER_TRANSLATION_JOINTS = [
    "gripper_joint_x",
    "gripper_joint_y",
    "gripper_joint_z",
]
UMI_GRIPPER_ROTATION_JOINTS = [
    "gripper_joint_rx",
    "gripper_joint_ry",
    "gripper_joint_rz",
]

UMI_GRIPPER_ARM_JOINTS = UMI_GRIPPER_TRANSLATION_JOINTS + UMI_GRIPPER_ROTATION_JOINTS

UMI_GRIPPER_FINGER_JOINTS = [
    "left_finger_joint", 
    "right_finger_joint"
]

UMI_GRIPPER_FINGER_BODY_NAMES = [
    "left_finger",
    "right_finger",
]

UMI_GRIPPER_ROOT_PRIM_PATH = "{ENV_REGEX_NS}/UmiGripper"
UMI_GRIPPER_BASE_FRAME_PRIM_PATH = f"{UMI_GRIPPER_ROOT_PRIM_PATH}/world"
UMI_GRIPPER_END_EFFECTOR_PRIM_PATH = f"{UMI_GRIPPER_ROOT_PRIM_PATH}/linear_rail"
UMI_GRIPPER_CAMERA_PRIM_PATH = f"{{ENV_REGEX_NS}}/UmiGripper/gopro/FisheyeCamera"

UMI_GRIPPER_FRAME_CFG = RobotFrameCfg(
    root_prim_path=UMI_GRIPPER_ROOT_PRIM_PATH,
    source_frame_prim_path=UMI_GRIPPER_BASE_FRAME_PRIM_PATH,
    end_effector_prim_path=UMI_GRIPPER_END_EFFECTOR_PRIM_PATH,
)

UMI_GRIPPER = ArticulationCfg(
    prim_path=UMI_GRIPPER_ROOT_PRIM_PATH,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_PATH}/umi_gripper_mirror/umi_gripper.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "gripper_joint_x": 0.0,
            "gripper_joint_y": 0.0,
            "gripper_joint_z": 0.0,
            "gripper_joint_rx": 0.0,
            "gripper_joint_ry": 0.0,
            "gripper_joint_rz": 0.0,
            "left_finger_joint": 0.0,
            "right_finger_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "translation": ImplicitActuatorCfg(
            joint_names_expr=UMI_GRIPPER_TRANSLATION_JOINTS,
            effort_limit_sim=200.0,
            velocity_limit_sim=2.0,
            stiffness=200.0,
            damping=10.0,
        ),
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=UMI_GRIPPER_ROTATION_JOINTS,
            effort_limit_sim=100.0,
            velocity_limit_sim=6.0,
            stiffness=120.0,
            damping=12.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=UMI_GRIPPER_FINGER_JOINTS,
            effort_limit_sim=40.0,
            velocity_limit_sim=10.0,
            stiffness=800.0,
            damping=10.0,
        ),
    },
)
