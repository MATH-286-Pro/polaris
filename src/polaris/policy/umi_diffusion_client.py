from collections import deque
from dataclasses import dataclass, field
import sys
from pathlib import Path

import torch
import numpy as np
from openpi_client import websocket_client_policy
from polaris.policy.abstract_client import InferenceClient, PolicyArgs
from isaaclab.envs import ManagerBasedRLEnvCfg

from .. import tool_linalg
from ..robot.robot_controller import UMI_Gripper_Controller
from ..environments.sensor.camera import GoPro_2_7K, FISHEYE_CAMERA_API

REPO_DIR = Path(__file__).resolve().parents[3]
ROBOT_UMI_DIR = REPO_DIR / "robot_umi_module"
sys.path.insert(0, str(ROBOT_UMI_DIR))
from diffusion_policy.common.cv2_util import get_image_transform  # type: ignore  # noqa: E402



# Pose/action codec used by the websocket API.
# Client sends EEF policy observations in IsaacSim convention. For relative
# observations, the client sends `_g`; the server converts IsaacSim convention
# to UMI/camera convention before policy inference.
class UMI_ACTION_API:
    # =========== Basic Linear Algebra Function ============== #
    # umi 本身 bug
    # 6D 使用的是旋转矩阵前两行 而不是前两列
    # 参考 issue: https://github.com/real-stanford/universal_manipulation_interface/issues/77

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
    def ACTION_10D_TO_TF_GRIPPER(cls, action_10d):
        """Convert a 10D pose action to TF and gripper-width command."""

        # Action Components (10D)
        # 3 pos
        # 6 rot
        # 1 gripper

        # Extract Action
        pos   = action_10d[..., :3]
        rot6d = action_10d[..., 3:9]
        rot   = cls.rot6d_to_rot(rot6d)  # note umi takes first 2 rows rather than first 2 columns
        gripper_width = action_10d[..., -1]

        # Build Transform Matrix
        tf = tool_linalg.pos_rot_to_tf(pos, rot)

        return tf, gripper_width




@dataclass
class UMIObservationHistory:
    obs_meta: dict
    obs_pose_repr: str
    max_horizon: int
    _history_b: deque[dict[str, np.ndarray]] = field(init=False)
    episode_start_eef_tf_b: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._history_b = deque(maxlen=self.max_horizon)

    def add(self, observation: dict[str, np.ndarray]) -> None:
        observation_b = {
            "camera0_rgb":          observation["camera0_rgb"],
            "robot0_eef_tf_b":      observation["robot0_eef_tf_b"],
            "robot0_gripper_width": observation["robot0_gripper_width"],
        }
        if self.episode_start_eef_tf_b is None:
            self.episode_start_eef_tf_b = observation_b["robot0_eef_tf_b"]
        self._history_b.append(observation_b)

    def clear(self) -> None:
        self._history_b.clear()
        self.episode_start_eef_tf_b = None

    @property
    def obs(self) -> dict[str, np.ndarray]:
        current_obs_b    = self._observation_window(1)[-1]
        current_eef_tf_b = current_obs_b["robot0_eef_tf_b"]

        start_eef_tf_b = (
            self.episode_start_eef_tf_b
            if self.episode_start_eef_tf_b is not None
            else current_eef_tf_b
        )

        request_obs: dict[str, np.ndarray] = {}

        for key, meta in self.obs_meta.items():
            if meta.get("ignore_by_policy", False):
                continue

            # OBS Information
            type    = meta["type"]   # "rgb", "low_dim"
            horizon = meta["horizon"]

            obs_list = self._observation_window(horizon)

            # OBS Process
            match type.lower():
                case "rgb":
                    request_obs[key] = np.stack(
                        [
                            self._format_image(obs[key], meta["shape"])
                            for obs in obs_list
                        ],
                        axis=0,
                    )

                case "low_dim":
                    key_lower = key.lower()
                    match key_lower:

                        # 多帧历史说明
                        # [-1] 为当前帧，[-2] 为上一帧，以此类推
                        # 所以 [-1] 的 pos = [0,0,0]

                        # 直接返回 width
                        case "robot0_gripper_width":
                            request_obs[key] = self._stack_observation_key(obs_list, key)

                        # 当前 tf_hist 相对于 current_tf 的表示
                        case "robot0_eef_pos":
                            eef_tf_b_hist = self._stack_observation_key(obs_list, "robot0_eef_tf_b")
                            eef_tf_g_hist = np.linalg.inv(current_eef_tf_b) @ eef_tf_b_hist
                            eef_pose9d_g_hist = UMI_ACTION_API.tf_to_pose9d(eef_tf_g_hist)
                            request_obs[key] = eef_pose9d_g_hist[..., 0:3]

                        # 当前 tf_hist 相对于 current_tf 的表示
                        case "robot0_eef_rot_axis_angle":
                            eef_tf_b_hist = self._stack_observation_key(obs_list, "robot0_eef_tf_b")
                            eef_tf_g_hist = np.linalg.inv(current_eef_tf_b) @ eef_tf_b_hist
                            eef_pose9d_g_hist = UMI_ACTION_API.tf_to_pose9d(eef_tf_g_hist)
                            request_obs[key] = eef_pose9d_g_hist[..., 3:9]

                        # 当前 tf_hist 相对于 start_tf 的表示
                        case "robot0_eef_rot_axis_angle_wrt_start":
                            eef_tf_b_hist = self._stack_observation_key(obs_list, "robot0_eef_tf_b")
                            eef_tf_s_hist = np.linalg.inv(start_eef_tf_b) @ eef_tf_b_hist
                            eef_pose9d_s_hist = UMI_ACTION_API.tf_to_pose9d(eef_tf_s_hist)
                            request_obs[key] = eef_pose9d_s_hist[..., 3:9]

                        case _:
                            raise KeyError(f"Unsupported UMI observation key: {key}")

                case _:
                    raise KeyError(f"Unsupported UMI observation type: {type}")
                

        return {key: value.astype(np.float32) for key, value in request_obs.items()}

    def _observation_window(self, horizon: int) -> list[dict[str, np.ndarray]]:
        observations = list(self._history_b)
        if len(observations) < horizon:
            pad_count = horizon - len(observations)
            observations = [observations[0]] * pad_count + observations
        return observations[-horizon:]


    @staticmethod
    def _stack_observation_key(
        obs_list: list[dict[str, np.ndarray]],
        key: str,
    ) -> np.ndarray:
        return np.stack([obs[key] for obs in obs_list], axis=0)

    @staticmethod
    def image_dimension_check(image: np.ndarray, shape: list[int]) -> None:
        channels, height, width = shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel image shape, got {shape}")
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image with 3 dimensions, got {image.shape}")
        if image.shape != (height, width, channels):
            raise ValueError(
                f"Expected image shape {(height, width, channels)} from shape_meta, got {image.shape}"
            )

    @staticmethod
    def _format_image(image: np.ndarray, shape: list[int]) -> np.ndarray:
        image = np.asarray(image)
        UMIObservationHistory.image_dimension_check(image, shape)

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return np.moveaxis(image, -1, 0)



@dataclass
class RealtimeTraj:

    def __init__(self, history_s: float = 2.0):
        self.history_s = float(history_s)
        self.time = np.empty((0,), dtype=np.float64)
        self.traj_tf_w = np.empty((0, 4, 4), dtype=np.float32)
        self.traj_grip_width = np.empty((0,), dtype=np.float32)

    @staticmethod
    def build_data(start_time: float, traj_tf_w, traj_grip_width, freq: float):

        # Check is 4 by 4 transform matrix
        if traj_tf_w.shape[-2:] != (4, 4):
            raise ValueError(f"traj_tf_w must have shape (..., 4, 4), got {traj_tf_w.shape}")
        
        # 机器人数据
        traj_tf_w = traj_tf_w.reshape(-1, 4, 4)
        traj_grip_width = np.asarray(traj_grip_width, dtype=np.float32).reshape(-1)

        # Build Time Sequence
        dt = 1 / freq
        time = start_time + np.arange(len(traj_tf_w), dtype=np.float64) * dt

        return {
            "time_seq": time,
            "traj_tf_w": traj_tf_w,
            "traj_grip_width": traj_grip_width,
        }

    def update(self, data):
        time_seq        = np.asarray(data["time_seq"], dtype=np.float64).reshape(-1)
        traj_tf_w       = np.asarray(data["traj_tf_w"], dtype=np.float32).reshape(-1, 4, 4)
        traj_grip_width = np.asarray(data["traj_grip_width"], dtype=np.float32).reshape(-1)

        if len(time_seq) == 0:
            return

        keep_end = np.searchsorted(self.time, time_seq[0], side="left")
        if keep_end > 0:
            self.time = np.concatenate([self.time[:keep_end], time_seq], axis=0)
            self.traj_tf_w = np.concatenate([self.traj_tf_w[:keep_end], traj_tf_w], axis=0)
            self.traj_grip_width = np.concatenate(
                [self.traj_grip_width[:keep_end], traj_grip_width],
                axis=0,
            )
        else:
            self.time = time_seq
            self.traj_tf_w = traj_tf_w
            self.traj_grip_width = traj_grip_width

        self._prune_history(time_seq[0])
    

    # 对于大于或小于当前数据 time 的 times 将取第一个或最后一个数据。
    def get_wbc_traj_w(self, times):

        times = np.asarray(times, dtype=np.float64)
        output_shape = times.shape
        query_time = np.clip(times.reshape(-1), self.time[0], self.time[-1])

        # 差值计算 tf
        pos = np.stack(
            [
                np.interp(query_time, self.time, self.traj_tf_w[:, axis, 3])
                for axis in range(3)
            ],
            axis=-1,
        )
        rot = self._interp_rotations(query_time, self.time, self.traj_tf_w[:, :3, :3])
        tf = tool_linalg.pos_rot_to_tf(pos, rot).astype(np.float32)
        return tf.reshape(output_shape + (4, 4))
    
    def get_wbc_grip(self, current_time):

        grip = np.interp(
            np.asarray(current_time, dtype=np.float64),
            self.time,
            self.traj_grip_width,
        )
        if isinstance(grip, np.ndarray):
            return grip.astype(np.float32)
        return np.float32(grip)

    def clear(self):
        self.time = np.empty((0,), dtype=np.float64)
        self.traj_tf_w = np.empty((0, 4, 4), dtype=np.float32)
        self.traj_grip_width = np.empty((0,), dtype=np.float32)

    def _prune_history(self, current_time: float):
        keep_start = np.searchsorted(
            self.time,
            float(current_time) - self.history_s,
            side="left",
        )
        if keep_start <= 0:
            return

        self.time = self.time[keep_start:]
        self.traj_tf_w = self.traj_tf_w[keep_start:]
        self.traj_grip_width = self.traj_grip_width[keep_start:]


    @staticmethod
    def _interp_rotations(target_time, time, rot):
        if len(time) == 1:
            return np.broadcast_to(rot[0], target_time.shape + (3, 3)).copy()

        right = np.searchsorted(time, target_time, side="right")
        right = np.clip(right, 1, len(time) - 1)
        left = right - 1

        denom = np.maximum(time[right] - time[left], 1e-12)
        alpha = ((target_time - time[left]) / denom)[..., None]

        q0 = tool_linalg._rot_to_quat(rot[left])
        q1 = tool_linalg._rot_to_quat(rot[right])
        same_hemisphere = np.sum(q0 * q1, axis=-1, keepdims=True) >= 0.0
        q1 = np.where(same_hemisphere, q1, -q1)

        dot = np.clip(np.sum(q0 * q1, axis=-1, keepdims=True), -1.0, 1.0)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        linear = sin_theta < 1e-6
        s0 = np.sin((1.0 - alpha) * theta) / np.maximum(sin_theta, 1e-12)
        s1 = np.sin(alpha * theta) / np.maximum(sin_theta, 1e-12)
        quat = np.where(linear, (1.0 - alpha) * q0 + alpha * q1, s0 * q0 + s1 * q1)
        quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
        return tool_linalg._quat_to_rot(quat)


@InferenceClient.register(client_name="UmiGripperPos")
class UmiGripperPosClient(InferenceClient):
    def __init__(self, args: PolicyArgs, env_cfg: ManagerBasedRLEnvCfg =None) -> None:
        self.args = args
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for UmiGripperPosClient")
        if env_cfg is None:
            raise ValueError("env_cfg must be passed when creating UmiGripperPosClient")

        self.client_policy = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port
        )
        self.actions_from_chunk_completed = 0
        self.action_10d_chunk_isc_e = None
        self.action_open_loop_horizon = args.open_loop_horizon
        self._needs_server_reset = True

        # Create Observation Buffer (High Level Policy)
        metadata = self.client_policy.get_server_metadata()
        self.shape_meta = metadata["shape_meta"]
        self.obs_meta = self.shape_meta["obs"]
        self.obs_pose_repr = metadata["obs_pose_repr"]
        self.action_pose_repr = metadata["action_pose_repr"]
        self.obs_horizons = metadata.get("obs_horizons", {})
        self.max_obs_horizon = max([1, *self.obs_horizons.values()])
        self.umi_obs_history = UMIObservationHistory(
            obs_meta=self.obs_meta,
            obs_pose_repr=self.obs_pose_repr,
            max_horizon=self.max_obs_horizon,
        )

        # Create Command Trajectory Buffer (High/Low Level Policy)
        self.realtime_traj = RealtimeTraj()

        # Create Controller (Low Level Policy)
        self.low_level_controller = UMI_Gripper_Controller()

        # 维护内部 step buffer 用于异步控制
        self.STEP = 0
        self.HIGH_LEVEL_TRAJ_FREQ = args.freq_traj_high #Hz
        self.LOW_LEVEL_FREQ  = args.freq_low #Hz
        self.ENV_FREQ = int(1.0 / (env_cfg.sim.dt * env_cfg.decimation))

        self.high_level_step_interval = int(self.ENV_FREQ / self.HIGH_LEVEL_TRAJ_FREQ)
        self.low_level_step_interval  = int(self.ENV_FREQ / self.LOW_LEVEL_FREQ)

        assert self.ENV_FREQ % self.HIGH_LEVEL_TRAJ_FREQ == 0
        assert self.ENV_FREQ % self.LOW_LEVEL_FREQ == 0

        print("Env Frequency = ", self.ENV_FREQ)
        print("High Level Frequency = ", self.HIGH_LEVEL_TRAJ_FREQ)
        print("Low Level Frequency = ", self.LOW_LEVEL_FREQ)

        self.action_robot    = None

    @property
    def rerender(self) -> bool:
        return False
        # return (
        #     self.actions_from_chunk_completed % self.open_loop_horizon == 0
        # )

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        return curr_obs["camera0_rgb"]

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.action_robot = None
        self._needs_server_reset = True
        self.STEP = 0
        self.realtime_traj.clear()
        self.umi_obs_history.clear()

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Infer the next action from the policy in a server-client setup
        """
        viz = None
        curr_obs_umi = self._extract_observation(obs)

        current_eef_tf_isc_b  = curr_obs_umi["eef_tf_isc_b"]
        current_eef_tf_isc_w  = curr_obs_umi["eef_tf_isc_w"]
        current_base_tf_isc_w = curr_obs_umi["base_tf_isc_w"]
        current_timestamp = float(obs["timestamp"])

        # ========================================== High Level Policy ========================================== #
        if self.STEP % self.high_level_step_interval == 0:
            self.umi_obs_history.add(curr_obs_umi)

            if (self.actions_from_chunk_completed % self.action_open_loop_horizon == 0):
                self.actions_from_chunk_completed = 0

                # 观测
                obs_isc_e = self.umi_obs_history.obs

                request_obs = {
                    "obs":           obs_isc_e,
                    "reset_episode": self._needs_server_reset,
                    # "prompt":       instruction,  # for VLA, umi doesn't need this
                }
                self._needs_server_reset = False

                # 推理
                server_response = self.client_policy.infer(request_obs)
                infer_ms = server_response["server_timing"]["infer_ms"]

                self.action_10d_chunk_isc_e = server_response["actions"]
                viz = curr_obs_umi["gopro"]

                target_traj_tf_isc_e, target_traj_gripper_width = UMI_ACTION_API.ACTION_10D_TO_TF_GRIPPER(self.action_10d_chunk_isc_e)
                target_traj_tf_isc_w    = current_eef_tf_isc_w[None, None, :, :] @ target_traj_tf_isc_e
                viz = self.visual_debug(
                    viz,
                    GoPro_2_7K,
                    target_traj_tf_isc_w,
                    current_eef_tf_isc_w,
                )

                # ============ 数据接口 ============= #                
                data = self.realtime_traj.build_data(
                    current_timestamp, 
                    target_traj_tf_isc_w, 
                    target_traj_gripper_width, 
                    self.HIGH_LEVEL_TRAJ_FREQ)
                
                self.realtime_traj.update(data)

            if return_viz and viz is None:
                viz = curr_obs_umi["gopro"]

            self.actions_from_chunk_completed += 1
            

        # ========================================== Low Level Policy ========================================== #
        if self.STEP % self.low_level_step_interval == 0:
            
            # 使用 buffer 数据
            target_tf_isc_w      = self.realtime_traj.get_wbc_traj_w(current_timestamp)
            target_gripper_width = self.realtime_traj.get_wbc_grip(current_timestamp)

            # 世界坐标 -> 体坐标 (Real Time 数据)
            current_world_tf_b = np.linalg.inv(current_base_tf_isc_w)
            target_tf_isc_b = current_world_tf_b @ target_tf_isc_w

            # LL WBC 推理
            action_robot = self.low_level_controller.tf_b_to_joint(
                target_tf_isc_b,
                target_gripper_width
                )
            
            self.action_robot = action_robot

            # 后处理 (在 ActionCfg 中已经处理过了)
            # self.action_robot = self.action_robot * action_scale + action_offset


        # 更新环境步
        self.STEP += 1
        self.STEP %= self.ENV_FREQ

        return self.action_robot, viz

    def _extract_observation(self, obs_dict) -> dict:

        # 处理 Image
        camera_rgb     = obs_dict["splat"].get("wrist_cam")
        input_height, input_width = camera_rgb.shape[:2]

        transform = get_image_transform(
            input_res=(input_width, input_height),
            output_res=(224,224),
            bgr_to_rgb=False,
        )
        camera_rgb_umi = transform(camera_rgb)

        # 详见 env_umi_cfg.py
        robot_state   = obs_dict["policy"]
        arm_joint     = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
        gripper_joint = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]
        
        # TODO Privilige Info (Need to delete later)
        eef_tf_isc_b_priv = robot_state["eef_tf_b_priv"].clone().detach().cpu().numpy()[0]
        eef_tf_isc_w_priv = robot_state["eef_tf_w_priv"].clone().detach().cpu().numpy()[0]
        base_tf_isc_w_priv = robot_state["base_tf_w_priv"].clone().detach().cpu().numpy()[0]
    
        return {
            "camera0_rgb":                     camera_rgb_umi,                                  #0000ff
            "robot0_gripper_width":            gripper_joint.astype(np.float32),                #0000ff #00ff00
            "robot0_eef_tf_b":                 eef_tf_isc_b_priv,                               #00ff00

            "eef_tf_isc_b":                    eef_tf_isc_b_priv,
            "eef_tf_isc_w":                    eef_tf_isc_w_priv,
            "base_tf_isc_w":                   base_tf_isc_w_priv,

            "arm_joint":                       arm_joint,
            "gopro":                           camera_rgb,
        }

    def visual_debug(self, img, cam_cfg, traj_tf_isc_w, current_eef_tf_isc_w):

        chw_image = img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4)
        img_hwc = np.moveaxis(img, 0, -1) if chw_image else img
        img_debug = np.ascontiguousarray(img_hwc.copy())

        k = cam_cfg.K # 注意这里默认 图片长宽与 cfg 一致

        current_world_tf_isc_e = np.linalg.inv(current_eef_tf_isc_w)
        traj_tf_isc_e_realtime = current_world_tf_isc_e[None, None, :, :] @ traj_tf_isc_w

        # 暂时命名
        eef_tf_isc_c = np.eye(4)
        eef_tf_isc_c[..., :3, 3] = np.array([+0.1922, 0, -0.09])
        traj_tf_isc_c_realtime = eef_tf_isc_c[None, None, :, :] @ traj_tf_isc_e_realtime

        # 转化 isc 到 camera convention
        traj_tf_cam_c_realtime = FISHEYE_CAMERA_API.TF_ISC_TO_CAM(traj_tf_isc_c_realtime)

        # 使用相机内参矩阵把 traj_tf_cam_e_realtime 转为 camera XY 二维坐标
        traj_xyz_cam_realtime = traj_tf_cam_c_realtime[..., :3, 3]
        traj_xy_cam_realtime, valid = FISHEYE_CAMERA_API.PROJECT_CAM_FISHEYE_XY(traj_xyz_cam_realtime)

        # 叠加 traj_xy_cam_realtime 点到 Image 上
        FISHEYE_CAMERA_API.DRAW_PROJECTED_POINTS(img_debug, traj_xy_cam_realtime, valid)

        return np.moveaxis(img_debug, -1, 0) if chw_image else img_debug
