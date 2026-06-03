import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math
import numpy as np
from typing import Sequence

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


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
