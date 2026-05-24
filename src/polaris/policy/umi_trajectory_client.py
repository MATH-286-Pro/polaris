import pickle
from pathlib import Path

import numpy as np

from polaris.policy.abstract_client import InferenceClient, PolicyArgs
from .. import tool_linalg


@InferenceClient.register(client_name="UmiTrajectory")
class UmiTrajectoryClient(InferenceClient):
    """Open-loop UMI controller that plays one trajectory from a pickle file."""

    def __init__(self, args: PolicyArgs) -> None:
        if args.trajectory_path is None:
            raise ValueError("trajectory_path must be set for UmiTrajectoryClient")
        if args.trajectory_stride < 1:
            raise ValueError("trajectory_stride must be >= 1")
        if args.trajectory_control_dt is not None and args.trajectory_control_dt <= 0.0:
            raise ValueError("trajectory_control_dt must be positive or None")

        self.args = args
        self.trajectory_path = Path(args.trajectory_path).expanduser()
        self.trajectory_index = args.trajectory_index
        self.trajectory_stride = args.trajectory_stride
        self.trajectory_control_dt = args.trajectory_control_dt
        self.actions = self._load_actions()
        self.step_idx = 0

    @property
    def rerender(self) -> bool:
        return False

    def reset(self):
        self.step_idx = 0

    def infer(
        self, obs, instruction, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        del instruction, return_viz

        action = self.actions[min(self.step_idx, len(self.actions) - 1)]
        self.step_idx += 1
        return action.copy(), self._visualize(obs)

    def _visualize(self, obs: dict) -> np.ndarray | None:
        splat_obs = obs.get("splat", {})
        if not splat_obs:
            return None

        image = splat_obs.get("wrist_cam")
        if image is None:
            image = next(iter(splat_obs.values()))

        return _resize_with_pad(image, 224, 224)

    def _load_actions(self) -> np.ndarray:
        with self.trajectory_path.open("rb") as f:
            data = pickle.load(f)

        if not isinstance(data, (list, tuple)):
            raise ValueError(
                f"Expected trajectory file to contain a list/tuple, got {type(data)}"
            )
        if not 0 <= self.trajectory_index < len(data):
            raise ValueError(
                f"trajectory_index {self.trajectory_index} is out of range for "
                f"{len(data)} trajectories"
            )

        trajectory = data[self.trajectory_index]
        if not isinstance(trajectory, dict):
            raise ValueError(
                f"Expected trajectory item to be a dict, got {type(trajectory)}"
            )

        pos, rot = self._trajectory_pose_to_pos_rot(trajectory)
        gripper_width = self._trajectory_gripper_width(trajectory, len(pos))
        t = self._trajectory_time(trajectory, len(pos))
        pos, rot, gripper_width = self._resample_to_control_time(
            pos, rot, gripper_width, t
        )

        # Trajectory positions are already in IsaacSim action convention:
        # gripper_joint_x/y/z directly. Only orientation needs URDF-chain
        # decomposition because rx/ry/rz are serial joints, not a generic API.
        joint_rot = tool_linalg.rot_to_euler_rad(rot).astype(np.float32)
        gripper = gripper_width.astype(np.float32)[:, None]
        return np.concatenate([pos.astype(np.float32), joint_rot, gripper], axis=-1)

    def _trajectory_pose_to_pos_rot(self, trajectory: dict) -> tuple[np.ndarray, np.ndarray]:
        if "tf" in trajectory:
            pos, rot = tool_linalg.tf_to_pos_rot(trajectory["tf"])
        else:
            missing = {"ee_pos", "ee_axis_angle"} - set(trajectory)
            if missing:
                raise ValueError(f"Trajectory is missing required keys: {sorted(missing)}")
            pos = np.asarray(trajectory["ee_pos"], dtype=float)
            rot = tool_linalg.axis_angle_to_rot(trajectory["ee_axis_angle"])

        if pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError(f"ee_pos must have shape (T, 3), got {pos.shape}")
        if rot.ndim != 3 or rot.shape[-2:] != (3, 3):
            raise ValueError(f"rotation must have shape (T, 3, 3), got {rot.shape}")
        if len(pos) != len(rot):
            raise ValueError(f"Position/rotation length mismatch: {len(pos)} vs {len(rot)}")

        return pos, rot

    def _trajectory_gripper_width(self, trajectory: dict, length: int) -> np.ndarray:
        if "gripper_width" not in trajectory:
            return np.zeros((length,), dtype=float)

        width = np.asarray(trajectory["gripper_width"], dtype=float)
        return width.reshape(length, -1)[:, 0]

    def _trajectory_time(self, trajectory: dict, length: int) -> np.ndarray | None:
        if "t" not in trajectory or self.trajectory_control_dt is None:
            return None

        t = np.asarray(trajectory["t"], dtype=float).reshape(-1)
        if len(t) != length:
            raise ValueError(f"Trajectory time length mismatch: {len(t)} vs {length}")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("Trajectory time values must be strictly increasing")
        return t - t[0]

    def _resample_to_control_time(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        gripper_width: np.ndarray,
        t: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if t is None:
            return (
                pos[:: self.trajectory_stride],
                rot[:: self.trajectory_stride],
                gripper_width[:: self.trajectory_stride],
            )

        target_t = np.arange(0.0, t[-1] + 1e-9, self.trajectory_control_dt)
        target_t = target_t[:: self.trajectory_stride]

        pos = np.stack(
            [np.interp(target_t, t, pos[:, axis]) for axis in range(3)], axis=-1
        )
        gripper_width = np.interp(target_t, t, gripper_width)
        rot = _interp_rotations(target_t, t, rot)
        return pos, rot, gripper_width

def _interp_rotations(target_t: np.ndarray, t: np.ndarray, rot: np.ndarray) -> np.ndarray:
    right = np.searchsorted(t, target_t, side="right")
    right = np.clip(right, 1, len(t) - 1)
    left = right - 1

    denom = np.maximum(t[right] - t[left], 1e-12)
    alpha = ((target_t - t[left]) / denom)[..., None]

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


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    image = np.asarray(image)
    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))

    y_idx = np.linspace(0, image_height - 1, resized_height).astype(np.int64)
    x_idx = np.linspace(0, image_width - 1, resized_width).astype(np.int64)
    resized = image[y_idx[:, None], x_idx[None, :]]

    if image.ndim == 2:
        output = np.zeros((height, width), dtype=image.dtype)
    else:
        output = np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    y0 = (height - resized_height) // 2
    x0 = (width - resized_width) // 2
    output[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return output
