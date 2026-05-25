import argparse
import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
import websockets
import websockets.frames
import websockets.server as websocket_server

from openpi_client import msgpack_numpy

REPO_DIR = Path(__file__).resolve().parents[3]
ROBOT_UMI_DIR = REPO_DIR / "robot_umi_module"
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(ROBOT_UMI_DIR))
from polaris import tool_linalg  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402


class CONVENTION_API:
    # Convention shift matrix.
    # UMI/camera +Z -> IsaacSim +X
    # UMI/camera +X -> IsaacSim -Y
    # UMI/camera +Y -> IsaacSim -X
    TF_SHIFT = np.array(
        [
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    TF_SHIFT_INV_T = np.linalg.inv(TF_SHIFT.T).astype(np.float32)

    assert np.allclose(TF_SHIFT.T, np.linalg.inv(TF_SHIFT))

    # UMI policy uses the first two rows of the rotation matrix for 6D rotation.
    # Reference: https://github.com/real-stanford/universal_manipulation_interface/issues/77
    @staticmethod
    def rot6d_to_rot(rot6d: np.ndarray) -> np.ndarray:
        a1 = rot6d[..., :3]
        a2 = rot6d[..., 3:6]

        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(min=1e-8)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(min=1e-8)
        b3 = np.cross(b1, b2)

        return np.stack([b1, b2, b3], axis=-2)

    @staticmethod
    def rot_to_rot6d(rot: np.ndarray) -> np.ndarray:
        return rot[..., :2, :].copy().reshape(rot.shape[:-2] + (6,)).astype(np.float32)

    @classmethod
    def tf_to_pose9d(cls, tf: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [tf[..., :3, 3], cls.rot_to_rot6d(tf[..., :3, :3])],
            axis=-1,
        ).astype(np.float32)

    @classmethod
    def pose9d_to_tf(cls, pose9d: np.ndarray) -> np.ndarray:
        pos = pose9d[..., :3]
        rot = cls.rot6d_to_rot(pose9d[..., 3:9])
        return tool_linalg.pos_rot_to_tf(pos, rot).astype(np.float32)

    @classmethod
    def TF_CAM_TO_ISC(cls, tf_umi: np.ndarray) -> np.ndarray:
        return (tf_umi @ cls.TF_SHIFT.T).astype(np.float32)

    @classmethod
    def TF_ISC_TO_CAM(cls, tf_isaacsim: np.ndarray) -> np.ndarray:
        return (tf_isaacsim @ cls.TF_SHIFT_INV_T).astype(np.float32)

    @classmethod
    def TF_CAM_RELATIVE_TO_ISC(cls, tf_cam_relative: np.ndarray) -> np.ndarray:
        """Convert a relative UMI transform into a relative IsaacSim transform."""

        return (cls.TF_SHIFT_INV_T @ tf_cam_relative @ cls.TF_SHIFT.T).astype(np.float32)

    @classmethod
    def TF_ISC_RELATIVE_TO_CAM(cls, tf_isc_relative: np.ndarray) -> np.ndarray:
        """Convert a relative IsaacSim transform into a relative UMI transform."""

        return (cls.TF_SHIFT.T @ tf_isc_relative @ cls.TF_SHIFT_INV_T).astype(np.float32)

    @classmethod
    def CAM_TO_ISC_ACTION_10D(cls, action_10d_cam: np.ndarray) -> np.ndarray:
        """Convert policy relative actions from UMI convention to IsaacSim convention."""

        action_10d_cam = np.asarray(action_10d_cam, dtype=np.float32)
        action_9d_cam  = action_10d_cam[..., :9]

        action_tf_cam = cls.pose9d_to_tf(action_9d_cam)
        action_tf_isc = cls.TF_CAM_RELATIVE_TO_ISC(action_tf_cam)

        action_9d_isc  = cls.tf_to_pose9d(action_tf_isc)
        action_gripper = action_10d_cam[..., 9:10]
        action_10d_isc = np.concatenate([action_9d_isc, action_gripper], axis=-1).astype(np.float32)

        return action_10d_isc

    @classmethod
    def ISC_TO_CAM_OBS(cls, obs_dict_isc_g: dict) -> dict[str, np.ndarray]:
        """Convert observation convention only: `obs_isc_g -> obs_cam_g`."""

        obs_dict_cam_g = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in obs_dict_isc_g.items()
        }
        cls._convert_pose_pair_isc_to_cam(
            obs_dict_cam_g,
            pos_key="robot0_eef_pos",
            rot_key="robot0_eef_rot_axis_angle",
        )
        cls._convert_pose_pair_isc_to_cam(
            obs_dict_cam_g,
            pos_key="robot0_eef_pos_wrt_start",
            rot_key="robot0_eef_rot_axis_angle_wrt_start",
        )
        return obs_dict_cam_g

    @classmethod
    def _convert_pose_pair_isc_to_cam(
        cls,
        obs_dict: dict[str, np.ndarray],
        pos_key: str,
        rot_key: str,
    ) -> None:
        if pos_key not in obs_dict and rot_key not in obs_dict:
            return

        pose9d_isc = cls._client_pose9d_isc(obs_dict, pos_key, rot_key)
        pose9d_cam = cls._pose9d_isc_g_to_cam_g(pose9d_isc)
        if pos_key in obs_dict:
            obs_dict[pos_key] = pose9d_cam[..., :3]
        if rot_key in obs_dict:
            obs_dict[rot_key] = pose9d_cam[..., 3:]

    @classmethod
    def _client_pose9d_isc(
        cls,
        obs_dict: dict,
        pos_key: str,
        rot_key: str,
    ) -> np.ndarray:
        pos = (
            np.asarray(obs_dict[pos_key], dtype=np.float32)
            if pos_key in obs_dict
            else None
        )
        rot6d = (
            np.asarray(obs_dict[rot_key], dtype=np.float32)
            if rot_key in obs_dict
            else None
        )

        if pos is None and rot6d is None:
            raise KeyError(
                f"Client observation is missing policy keys: {pos_key}, {rot_key}"
            )
        if pos is None:
            pos = np.zeros(rot6d.shape[:-1] + (3,), dtype=np.float32)
        if rot6d is None:
            identity_rot6d = cls.rot_to_rot6d(np.eye(3, dtype=np.float32))
            rot6d = np.broadcast_to(identity_rot6d, pos.shape[:-1] + (6,))

        return np.concatenate([pos, rot6d], axis=-1).astype(np.float32)

    @classmethod
    def _pose9d_isc_g_to_cam_g(cls, pose9d_isc_g: np.ndarray) -> np.ndarray:
        """Convert relative pose9d convention only: `isc_g -> cam_g`."""

        pose_tf_isc_g = cls.pose9d_to_tf(pose9d_isc_g)
        pose_tf_cam_g = cls.TF_ISC_RELATIVE_TO_CAM(pose_tf_isc_g)
        return cls.tf_to_pose9d(pose_tf_cam_g)


class UmiDiffusionPolicy:
    def __init__(
        self,
        checkpoint: str,
        device: str,
    ) -> None:
        
        # The following loading precedure is the same as "umi/eval_real_umi.py"

        # Load check point
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, weights_only=False, map_location=device)

        # Extract model configuration
        cfg = payload["cfg"]

        # ? Maybe Pretrained Visual Encoder
        if OmegaConf.select(cfg, "policy.obs_encoder.pretrained") is not None:
            with open_dict(cfg):
                cfg.policy.obs_encoder.pretrained = False

        # Load Diffusion Policy
        cls = hydra.utils.get_class(cfg._target_)
        self.workspace = cls(cfg)
        self.workspace: BaseWorkspace
        self.workspace.load_payload(payload)

        policy = self.workspace.model
        if cfg.training.use_ema:
            policy = self.workspace.ema_model
        policy.num_inference_steps = 16 #00ff00 Hard Coding
        self.obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr: str = cfg.task.pose_repr.action_pose_repr  # = "relative"


        policy.eval().to(device)
        policy.reset()

        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(device)
        self.shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
        self.obs_meta = self.shape_meta["obs"]
        self.max_horizon = max(int(v.get("horizon", 1)) for v in self.obs_meta.values())

    @property
    def metadata(self) -> dict:
        return {
            "policy": "umi_diffusion_policy",
            "obs_keys": list(self.obs_meta.keys()),
            "obs_horizons": {
                key: int(meta.get("horizon", 1))
                for key, meta in self.obs_meta.items()
                if not meta.get("ignore_by_policy", False)
            },
            "max_obs_horizon":  self.max_horizon,
            "action_shape":     self.shape_meta["action"]["shape"],
            "obs_pose_repr":    self.obs_pose_repr,
            "action_pose_repr": self.action_pose_repr,

            "shape_meta": self.shape_meta,

            # Policy Input Output
            "task": {
                "low_dim_obs_horizon": self.cfg.task.low_dim_obs_horizon,
                "img_obs_horizon":     self.cfg.task.img_obs_horizon,
                "action_horizon":      self.cfg.task.action_horizon,
            }
        }

    def reset(self) -> None:
        self.policy.reset()

    def _align_policy_observation_horizon(
        self,
        obs_dict_cam_g: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        policy_obs: dict[str, np.ndarray] = {}

        for key, meta in self.obs_meta.items():
            if meta.get("ignore_by_policy", False):
                continue
            if key not in obs_dict_cam_g:
                raise KeyError(f"Client observation is missing policy key: {key}")

            horizon = int(meta.get("horizon", 1))
            obs_shape = tuple(int(dim) for dim in meta.get("shape", ()))
            value = np.asarray(obs_dict_cam_g[key], dtype=np.float32)
            policy_obs[key] = self._align_observation_horizon(
                value=value,
                horizon=horizon,
                obs_shape=obs_shape,
            )

        return {key: value.astype(np.float32) for key, value in policy_obs.items()}

    @staticmethod
    def _align_observation_horizon(
        value: np.ndarray,
        horizon: int,
        obs_shape: tuple[int, ...],
    ) -> np.ndarray:
        if value.ndim == 0:
            value = value.reshape((1,) + obs_shape) if obs_shape else value.reshape(1)
        elif obs_shape and value.shape == obs_shape:
            value = value[None, ...]
        elif obs_shape == (1,) and value.ndim == 1 and value.shape[0] == horizon:
            value = value[:, None]

        if len(value) < horizon:
            pad_count = horizon - len(value)
            pad = np.broadcast_to(value[0:1], (pad_count, *value.shape[1:]))
            value = np.concatenate([pad, value], axis=0)

        return value[-horizon:]

    def infer(self, request_obs: dict) -> dict:
        if request_obs.get("reset_episode", False):
            self.reset()

        # Note "robot0_eef_rot_axis_angle_wrt_start" is the eef_rot_hist w.r.t. the start eef_rot
        # Which is not _g, but _s !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        obs_dict_isc_g = request_obs.get("obs", request_obs)
        obs_dict_cam_g = CONVENTION_API.ISC_TO_CAM_OBS(obs_dict_isc_g)
        obs_dict_cam_g_policy = self._align_policy_observation_horizon(obs_dict_cam_g)
        obs_dict_cam_g_torch = {
            key: torch.from_numpy(np.asarray(value)).unsqueeze(0).to(self.device)
            for key, value in obs_dict_cam_g_policy.items()
        }

        with torch.no_grad():
            result = self.policy.predict_action(obs_dict_cam_g_torch)

        # 将 UMI frame convention 转为 IsaacSim convention
        # action 仍然是相对于 camera 的相对动作
        # 变量说明
        #   cam_g = camera frame convention + relative to gripper
        #   isc_g = isaacsim frame convention + relative to gripper
        action_10d_chunck_cam_g_tensor = result["action_pred"]
        action_10d_chunck_cam_g = action_10d_chunck_cam_g_tensor[0].detach().cpu().numpy()
        action_10d_chunck_isc_g = CONVENTION_API.CAM_TO_ISC_ACTION_10D(action_10d_chunck_cam_g)

        return {"actions": action_10d_chunck_isc_g}


class WebsocketPolicyServer:
    def __init__(self, policy: UmiDiffusionPolicy, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))

        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                start = time.monotonic()
                response = self.policy.infer(request)
                response["server_timing"] = {"infer_ms": (time.monotonic() - start) * 1000}
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a trained UMI diffusion policy over websocket.")
    parser.add_argument("--checkpoint", required=True, help="Path to diffusion policy .ckpt file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on.")
    parser.add_argument("--device", default="cuda", help="Torch device for inference.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    policy = UmiDiffusionPolicy(
        checkpoint=args.checkpoint,
        device=args.device,
    )
    logging.info("Serving UMI diffusion policy on %s:%s", args.host, args.port)

    server = WebsocketPolicyServer(
        policy=policy, 
        host=args.host, 
        port=args.port
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
