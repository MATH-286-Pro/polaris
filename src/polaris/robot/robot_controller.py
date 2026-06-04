import torch
import numpy as np
from abc import ABC, abstractmethod

from .robot_cfg import UMI_EE_TO_JOINT_ORIGIN_B

from ..tool_linalg import (
    tf_to_pos_rot,
    rot_to_euler_rad,
)


class Abstract_Controller(ABC):

    M: np.array = None

    SLIST: np.array = None

    @classmethod
    @abstractmethod
    def joint_to_tf_b(cls, joints):
        """
        Input robot joint controll
        Out put transform matrix, gripper width
        """
        pass

    @classmethod
    @abstractmethod
    def tf_b_to_joint(cls, tf, gripper_width):
        """
        Input transform matrix, gripper width
        Out put robot joint controll

        This function can be a model-based controller
        Or a RL-based controller
        """
        pass


# ================== UMI Gripper Robot ====================== #
class UMI_Gripper_Controller(Abstract_Controller):
    """Controller for UMI gripper URDF joint commands."""

    M = np.eye(4, dtype=float)
    M[:3, 3] = UMI_EE_TO_JOINT_ORIGIN_B
    M_INV = np.eye(4, dtype=float)
    M_INV[:3, 3] = -UMI_EE_TO_JOINT_ORIGIN_B

    SLIST = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # gripper_joint_x
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # gripper_joint_y
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # gripper_joint_z
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # gripper_joint_rx
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # gripper_joint_ry
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # gripper_joint_rz
        ],
        dtype=float,
    ).T


    @classmethod
    def joint_to_tf_b(cls, joints):
        raise NotImplementedError

    @classmethod
    def tf_b_to_joint(cls, tf_b, gripper_width=None) -> np.ndarray:
        """Convert a target URDF-frame TF to UMI gripper joint-position commands.

        The UMI gripper URDF does not expose a single Euler-orientation command.
        Its pose is controlled by a serial chain of joints:

            T = T_xyz([0, 0, 0.11] + [q_x, q_y, q_z])
                @ Rx(q_rx) @ Ry(q_ry) @ Rz(q_rz)

        This function decomposes a desired transform for the ``linear_rail`` /
        ``base_link`` frame into the corresponding joint commands in
        ``UMI_GRIPPER_JOINT_NAMES`` order. The returned orientation values are
        joint angles for the URDF chain, not a generic roll-pitch-yaw API.
        """

        tf_base_b = tf_b
        tf_gripper_b = tf_base_b @ cls.M_INV

        pos_gripper_b, rot_gripper_b = tf_to_pos_rot(tf_gripper_b)
        joint_pos = pos_gripper_b
        joint_rot = rot_to_euler_rad(rot_gripper_b)

        command   = np.concatenate([joint_pos, joint_rot], axis=-1).astype(np.float32)

        gripper_width = np.asarray(gripper_width, dtype=np.float32)
        if gripper_width.shape == command.shape[:-1]:
            gripper_width = gripper_width[..., None]
        gripper_width = np.broadcast_to(
            gripper_width, command.shape[:-1] + (1,)
        )
        
        return np.concatenate([command, gripper_width], axis=-1)
    

# ================== A2-VX300S Robot ====================== #
class WBC_Controller(Abstract_Controller):
    """Controller for A2-VX300S robot using a low-level WBC policy."""

    def __init__(self, policy_path: str = None, device: str = "cuda"):
        from polaris.robot.load_rsl_policy import RSLPolicy
        self.policy = RSLPolicy(model_path=policy_path, device=device)
        self.device = device

    @staticmethod
    def joint_to_tf_b(joints):
        """
        Convert joint positions to end-effector transform.
        Note: For WBC, this usually requires forward kinematics of the specific arm.
        """
        # This is a placeholder for the FK logic if needed for the A2-VX300S arm
        raise NotImplementedError("FK for A2-VX300S not implemented in this controller.")

    def tf_b_to_joint(self, obs: torch.Tensor, gripper_width: torch.Tensor) -> np.array:
        """
        Input: WBC observation tensor
        Output: Joint position actions from the neural WBC policy
        """
        action_wbc = self.policy(obs)

        action_gripper_width = torch.as_tensor(
            gripper_width,
            dtype=action_wbc.dtype,
            device=action_wbc.device,
        ).reshape(-1, 1)

        action_robot = torch.cat([
            action_wbc, 
            action_gripper_width
            ], dim=-1)

        return action_robot.cpu().numpy()
