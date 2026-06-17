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
