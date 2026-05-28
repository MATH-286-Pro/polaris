import torch
from pathlib import Path
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math
import isaaclab.envs.mdp as mdp
import numpy as np
from typing import Sequence

from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab.utils import configclass, noise
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

import dataclasses
from dataclasses import MISSING
from ..robot.robot_cfg import RobotFrameCfg

### SceneCfg ###
@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    robot: ArticulationCfg = MISSING

    wrist_cam: CameraCfg = MISSING

    ee_frame: FrameTransformerCfg = MISSING

    sphere_light = AssetBaseCfg(
        prim_path="/World/biglight",
        spawn=sim_utils.DomeLightCfg(intensity=1000),
    )

    def make_ee_frame_cfg(self, frame_cfg: RobotFrameCfg) -> FrameTransformerCfg:
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame = FrameTransformerCfg(
            prim_path=frame_cfg.source_frame_prim_path,
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=frame_cfg.end_effector_prim_path,
                    name=frame_cfg.end_effector_name,
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

    def dynamic_setup(self, environment_path, robot_splat=True, nightmare="", **kwargs):
        environment_path_ = Path(environment_path)
        environment_path = str(environment_path_.resolve())

        scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=environment_path,
                activate_contact_sensors=False,
            ),
        )
        self.scene = scene
        if not robot_splat:
            self.robot.spawn.semantic_tags = [("class", "raytraced")]
        stage = Usd.Stage.Open(environment_path)
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()

        for child in children:
            name = child.GetName()
            print(name)

            # if its a camera, use the camera pose
            if child.IsA(UsdGeom.Camera):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (
                    rot.GetReal(),
                    rot.GetImaginary()[0],
                    rot.GetImaginary()[1],
                    rot.GetImaginary()[2],
                )
                asset = CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    height=720,
                    width=1280,
                    data_types=["rgb", "semantic_segmentation"],
                    colorize_semantic_segmentation=False,
                    spawn=None,
                    offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
                )
                setattr(self, name, asset)
            elif UsdPhysics.RigidBodyAPI(child):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (
                    rot.GetReal(),
                    rot.GetImaginary()[0],
                    rot.GetImaginary()[1],
                    rot.GetImaginary()[2],
                )
                asset = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=pos,
                        rot=rot,
                    ),
                )
                setattr(self, name, asset)

        if not hasattr(self, "external_cam"):
            self.external_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/scene/external_cam",
                height=720,
                width=1280,
                data_types=["rgb", "semantic_segmentation"],
                colorize_semantic_segmentation=False,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=1.0476,
                    horizontal_aperture=2.5452,
                    vertical_aperture=1.4721,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(-0.01, -0.33, 0.48),
                    rot=(0.76, 0.43, -0.24, -0.42),
                    convention="opengl",
                ),
            )


@configclass
class ActionCfg:
    pass


@configclass
class ObservationCfg:
    pass


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum configuration."""



class BasicEnvCfg(ManagerBasedRLEnvCfg):

    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()

    actions = ActionCfg()

    rewards = RewardsCfg()

    terminations = TerminationsCfg()

    commands = CommandsCfg()

    events = EventCfg()

    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 30
        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 4 * 2
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 4 * 2

        self.rerender_on_reset = True

    def dynamic_setup(self, *args):
        self.scene.dynamic_setup(*args)