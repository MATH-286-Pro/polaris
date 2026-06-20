import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math
import numpy as np
from typing import Sequence
import math as python_math


from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ManagerBasedEnv


def set_material_mirror(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Mirrir Parameter
    color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    metallic: float = 1.0
    roughness: float = 0.0

    del env_ids

    robot = env.scene[asset_cfg.name]
    stage = sim_utils.get_current_stage()
    material_path = "/World/Materials/material_mirror"
    shader_path = f"{material_path}/Shader"

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    if isinstance(asset_cfg.body_ids, slice):
        body_names = robot.body_names[asset_cfg.body_ids]
    else:
        body_names = [robot.body_names[body_id] for body_id in asset_cfg.body_ids]

    body_prims = []
    root_prims = sim_utils.find_matching_prims(robot.cfg.prim_path, stage=stage)

    for root_prim in root_prims:
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == "visuals" and prim.IsInstanceable():
                prim.SetInstanceable(False)

    for root_prim in root_prims:
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() in body_names:
                body_prims.append(prim)

    if not body_prims:
        print(
            f"Could not find UMI mirror prims {tuple(body_names)} "
            f"under '{robot.cfg.prim_path}'. "
        )
        return

    for body_prim in body_prims:
        for prim in Usd.PrimRange(body_prim):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        UsdShade.MaterialBindingAPI.Apply(body_prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        for prim in Usd.PrimRange(body_prim):
            if prim.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )


def set_material_friction(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    static_friction: float = 2.0,
    dynamic_friction: float = 2.0,
    friction_combine_mode: str = "average",
):
    del env_ids

    robot = env.scene[asset_cfg.name]
    stage = sim_utils.get_current_stage()
    material_path = "/World/Materials/material_friction"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        friction_combine_mode=friction_combine_mode,
    )
    material_cfg.func(material_path, material_cfg)

    collision_root_paths = []
    fallback_collision_prim_paths = []
    root_prims = sim_utils.find_matching_prims(robot.cfg.prim_path, stage=stage)
    for root_prim in root_prims:
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() in asset_cfg.body_names:
                for child in Usd.PrimRange(prim):
                    if child.GetName() == "collisions":
                        collision_root_paths.append(child.GetPath().pathString)
                    elif child.HasAPI(UsdPhysics.CollisionAPI):
                        fallback_collision_prim_paths.append(child.GetPath().pathString)

    prim_paths = collision_root_paths or fallback_collision_prim_paths
    if not prim_paths:
        raise RuntimeError(
            f"Could not find collision prims under {asset_cfg.body_names}"
            f"under '{robot.cfg.prim_path}'. "
            f"Available body names: {robot.body_names}"
        )

    for prim_path in prim_paths:
        sim_utils.bind_physics_material(prim_path, material_path, stage=stage)


def set_passive_finger_joint_limits(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lower: float = 0.0,
    upper: float = 0.02,
    joint_names: Sequence[str] | None = None,
):
    robot = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(
        joint_names,
        preserve_order=True,
    )

    num_envs = robot.num_instances if env_ids is None else len(env_ids)
    limits = torch.tensor(
        [lower, upper],
        dtype=torch.float32,
        device=robot.device,
    ).reshape(1, 1, 2)
    limits = limits.repeat(num_envs, len(joint_ids), 1)

    robot.write_joint_position_limit_to_sim(
        limits,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )



# ================================ VX300S arm parameters ================================ #
VX300S_ARM_JOINT_NAMES = (
    "waist", 
    "shoulder", 
    "elbow", 
    "forearm_roll", 
    "wrist_angle", 
    "wrist_rotate"
    )

#   elbow
#  __ _______ link2
# |
# | link1
# |
# - shoulder (yaw)
# |
# | link0
_VX300S_LINK0_OFFSET = [0,    0, 0.127]
_VX300S_LINK1_OFFSET = [0.06, 0, 0.3]
_VX300S_LINK2_OFFSET = [0.3,  0, 0]

_VX300S_LINK_1_LENGTH = python_math.hypot(_VX300S_LINK1_OFFSET[0], _VX300S_LINK1_OFFSET[2])
_VX300S_LINK_2_LENGTH = abs(_VX300S_LINK2_OFFSET[0])

_VX300S_LINK_1_HOME_ANGLE = python_math.atan2(_VX300S_LINK1_OFFSET[0], _VX300S_LINK1_OFFSET[2])
_VX300S_LINK_2_HOME_ANGLE = -_VX300S_LINK_1_HOME_ANGLE


# ================================ Tools for VX300S arm sampling and IK ================================ #
def _vx300s_ik(target_pos: torch.Tensor, default_joint_pos: torch.Tensor) -> torch.Tensor:
    """Fast position-only IK for waist, shoulder, and elbow.

    The wrist joints stay at their default values. Targets outside the 2-link
    planar workspace are projected to the nearest reachable radius.
    """
    joint_pos = default_joint_pos.clone()

    yaw = torch.atan2(target_pos[:, 1], target_pos[:, 0])
    x = torch.linalg.norm(target_pos[:, :2], dim=1)
    z = target_pos[:, 2] - _VX300S_LINK0_OFFSET[2]

    l1 = _VX300S_LINK_1_LENGTH
    l2 = _VX300S_LINK_2_LENGTH
    eps = torch.finfo(default_joint_pos.dtype).eps
    l3 = torch.sqrt(x.square() + z.square())
    l3 = torch.clamp(l3, min=abs(l1 - l2) + eps, max=l1 + l2 - eps) # prevent exceed sphere radius

    theta_m1 = torch.atan2(z, x)
    theta_m2 = torch.acos(torch.clamp((l1**2 + l3.square() - l2**2) / (2.0 * l1 * l3), -1.0, 1.0))
    theta_m3 = torch.acos(torch.clamp((l1**2 + l2**2 - l3.square()) / (2.0 * l1 * l2), -1.0, 1.0))

    # 计算角度
    theta_1 = python_math.pi / 2.0 - theta_m1 - theta_m2
    theta_2 = python_math.pi / 2.0 - theta_m3

    # 转换为 joint
    joint_pos[:, 0] = yaw
    joint_pos[:, 1] = theta_1 - _VX300S_LINK_1_HOME_ANGLE # - default_joint_pos[:, 1]
    joint_pos[:, 2] = theta_2 - _VX300S_LINK_2_HOME_ANGLE # - default_joint_pos[:, 2]

    return joint_pos


def _sample_vx300s_sector(
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    yaw_range: tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
    uniform_area: bool,
) -> torch.Tensor:
    rand = torch.rand((len(env_ids), 3), device=device, dtype=dtype)

    z_pos = rand[:, 1] * (z_range[1] - z_range[0]) + z_range[0]
    yaw = rand[:, 2] * (yaw_range[1] - yaw_range[0]) + yaw_range[0]

    if uniform_area:
        radius_min_sq = x_range[0] ** 2
        radius_max_sq = x_range[1] ** 2
        radius = torch.sqrt(rand[:, 0] * (radius_max_sq - radius_min_sq) + radius_min_sq)
    else:
        radius = rand[:, 0] * (x_range[1] - x_range[0]) + x_range[0]  # range from x[0] to x[1]

    target_pos = torch.zeros((len(env_ids), 3), device=device, dtype=dtype)
    target_pos[:, 0] = radius * torch.cos(yaw)
    target_pos[:, 1] = radius * torch.sin(yaw)
    target_pos[:, 2] = z_pos

    return target_pos

# ================================ Reset Functions ================================ #
def reset_vx300s_arm_by_ee_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (0.3, 0.4),
    z_range: tuple[float, float] = (-0.3, 0.1),
    yaw_range: tuple[float, float] = (-1.30899694, 1.30899694),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: Sequence[str] = VX300S_ARM_JOINT_NAMES,
    uniform_area: bool = True,
):
    """Reset VX300S arm joints by sampling an EE position in the arm-base sector.

    The sampled target is expressed in the VX300S space frame used by the provided
    product-of-exponentials model. ``x_range`` is treated as radial distance before
    rotating by ``yaw_range`` around +Z:
    ``target = [x * cos(yaw), x * sin(yaw), z]``.
    """
    
    asset: Articulation = env.scene[asset_cfg.name]

    # cache for acceleration
    joint_cache_name = f"_vx300s_reset_joint_ids_{asset_cfg.name}_{'_'.join(joint_names)}"
    joint_ids = getattr(env, joint_cache_name, None)
    if joint_ids is None:
        joint_ids = asset.find_joints(list(joint_names))[0]
        setattr(env, joint_cache_name, joint_ids)

    default_joint_pos = asset.data.default_joint_pos[env_ids][:, joint_ids]
    joint_vel = asset.data.default_joint_vel[env_ids][:, joint_ids].clone()

    # sample EE position for reset
    target_pos = _sample_vx300s_sector(
        env_ids=env_ids,
        x_range=x_range,
        z_range=z_range,
        yaw_range=yaw_range,
        device=default_joint_pos.device,
        dtype=default_joint_pos.dtype,
        uniform_area=uniform_area,
    )

    joint_pos = _vx300s_ik(target_pos, default_joint_pos)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)
