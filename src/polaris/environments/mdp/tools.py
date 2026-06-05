import torch
import math
import isaaclab.utils.math as math_utils

# rotation matrix -> quaternions
def rot_2_quat(rot: torch.Tensor):
    return math_utils.quat_from_matrix(rot)

# quaternions -> rotation matrix
def quat_2_rot(quat: torch.Tensor):
    return math_utils.matrix_from_quat(quat)

# transform matrix -> position + rotation matrix
def tf_2_pos_rot(tf: torch.Tensor):
    pos = tf[..., :3, 3]
    rot = tf[..., :3, :3]
    return pos, rot

# position + rotation matrix -> transform matrix
def pos_rot_2_tf(pos: torch.Tensor, rot: torch.Tensor):
    tf = torch.eye(4, dtype=pos.dtype, device=pos.device).expand(pos.shape[:-1] + (4, 4)).clone()
    tf[..., :3, 3]  = pos
    tf[..., :3, :3] = rot
    return tf

# [pos, quat] -> transform matrix
def pose_2_tf(pose: torch.Tensor):
    pos  = pose[..., :3]
    quat = pose[..., 3:7]
    rot  = math_utils.matrix_from_quat(quat)  # rotation matirx

    return pos_rot_2_tf(pos, rot)

# rot -> axis_angle [..., 3]
def rot_2_axis_angle(rot: torch.Tensor):
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (..., 3, 3), got {rot.shape}.")
    quat = rot_2_quat(rot)
    return math_utils.axis_angle_from_quat(quat)
