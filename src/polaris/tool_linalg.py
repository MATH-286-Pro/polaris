import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# ======= Base Function ======= #
@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero sub-gradient where x is 0.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L91-L99
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L102-L161
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )

@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# ======= Tool Function ======= #
def quat_to_rot(quat):
    match quat:
        case np.ndarray():
            quat = quat[:, [1, 2, 3, 0]]  # w x y z
            rot = R.from_quat(quat).as_matrix()
        case torch.Tensor():
            rot = matrix_from_quat(quat)

        case _:
            raise ValueError(f"quat must be np.ndarray or torch.Tensor, got {type(quat)}")
        
    return rot

def rot_to_quat(rot):
    match rot:
        case np.ndarray():
            quat = R.from_matrix(rot).as_quat()
            quat = quat[:, [3, 0, 1, 2]]  # w x y z

        case torch.Tensor():
            quat = quat_from_matrix(rot)
        
        case _:
            raise ValueError(f"rot must be np.ndarray or torch.Tensor, got {type(rot)}")
        
    return quat


def pos_rot_to_tf(pos, rot):

    # if pos.ndim != rot.ndim:
    #     raise ValueError(f"pos and rot must have the same number of dimensions, got {pos.ndim} and {rot.ndim}")

    match pos:
        case np.ndarray():
            pos = np.asarray(pos, dtype=float)
            rot = np.asarray(rot, dtype=float)

            if pos.shape[-1] != 3:
                raise ValueError(f"pos must have shape (..., 3), got {pos.shape}")
            if rot.shape[-2:] != (3, 3):
                raise ValueError(f"rot must have shape (..., 3, 3), got {rot.shape}")

            tf = np.broadcast_to(np.eye(4), pos.shape[:-1] + (4, 4)).copy()
            tf[..., :3, :3] = rot
            tf[..., :3, 3] = pos
            return tf

        case torch.Tensor():
            tf = torch.eye(4, dtype=pos.dtype, device=pos.device).expand(pos.shape[:-1] + (4, 4)).clone()
            tf[..., :3, 3]  = pos
            tf[..., :3, :3] = rot
            return tf
        
        case _:
            raise ValueError(f"pos and rot must be np.ndarray or torch.Tensor, got {type(pos)} and {type(rot)}")



def tf_to_pos_rot(tf):
    tf = np.asarray(tf, dtype=float)

    if tf.shape[-2:] != (4, 4):
        raise ValueError(f"tf must have shape (..., 4, 4), got {tf.shape}")

    pos = tf[..., :3, 3]
    rot = tf[..., :3, :3]
    return pos, rot


def axis_angle_to_rot(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=float)

    if axis_angle.shape[-1] != 3:
        raise ValueError(f"axis_angle must have shape (..., 3), got {axis_angle.shape}")

    angle = np.linalg.norm(axis_angle, axis=-1)
    safe_angle = np.where(angle < 1e-6, 1.0, angle)
    axis = axis_angle / safe_angle[..., None]
    x, y, z = np.moveaxis(axis, -1, 0)

    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.empty(axis_angle.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = c + x*x*C
    R[..., 0, 1] = x*y*C - z*s
    R[..., 0, 2] = x*z*C + y*s
    R[..., 1, 0] = y*x*C + z*s
    R[..., 1, 1] = c + y*y*C
    R[..., 1, 2] = y*z*C - x*s
    R[..., 2, 0] = z*x*C - y*s
    R[..., 2, 1] = z*y*C + x*s
    R[..., 2, 2] = c + z*z*C

    R[angle < 1e-6] = np.eye(3)
    return R


def rot_to_axis_angle(rot):
    rot = np.asarray(rot, dtype=float)

    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rot must have shape (..., 3, 3), got {rot.shape}")

    m00 = rot[..., 0, 0]
    m01 = rot[..., 0, 1]
    m02 = rot[..., 0, 2]
    m10 = rot[..., 1, 0]
    m11 = rot[..., 1, 1]
    m12 = rot[..., 1, 2]
    m20 = rot[..., 2, 0]
    m21 = rot[..., 2, 1]
    m22 = rot[..., 2, 2]

    q_abs = np.sqrt(
        np.maximum(
            np.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                axis=-1,
            ),
            0.0,
        )
    )

    quat_candidates = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], axis=-1),
            np.stack([m10 - m01, m02 + m20, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )
    quat_candidates = quat_candidates / (2.0 * np.maximum(q_abs[..., None], 1e-8))

    best = np.argmax(q_abs, axis=-1)
    quat = np.take_along_axis(quat_candidates, best[..., None, None], axis=-2)[..., 0, :]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)

    vector = quat[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, quat[..., 0])
    scale = angle / np.where(vector_norm < 1e-8, 1.0, vector_norm)
    scale = np.where(vector_norm < 1e-8, 2.0, scale)

    return vector * scale[..., None]


def _rot_to_quat(rot):
    rot = np.asarray(rot, dtype=float)

    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rot must have shape (..., 3, 3), got {rot.shape}")

    m00 = rot[..., 0, 0]
    m01 = rot[..., 0, 1]
    m02 = rot[..., 0, 2]
    m10 = rot[..., 1, 0]
    m11 = rot[..., 1, 1]
    m12 = rot[..., 1, 2]
    m20 = rot[..., 2, 0]
    m21 = rot[..., 2, 1]
    m22 = rot[..., 2, 2]

    q_abs = np.sqrt(
        np.maximum(
            np.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                axis=-1,
            ),
            0.0,
        )
    )

    quat_candidates = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], axis=-1),
            np.stack([m10 - m01, m02 + m20, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )
    quat_candidates = quat_candidates / (2.0 * np.maximum(q_abs[..., None], 1e-8))

    best = np.argmax(q_abs, axis=-1)
    quat = np.take_along_axis(quat_candidates, best[..., None, None], axis=-2)[..., 0, :]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    return np.where(quat[..., :1] < 0.0, -quat, quat)


def _quat_to_rot(quat):
    quat = np.asarray(quat, dtype=float)

    if quat.shape[-1] != 4:
        raise ValueError(f"quat must have shape (..., 4), got {quat.shape}")

    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = np.moveaxis(quat, -1, 0)

    rot = np.empty(quat.shape[:-1] + (3, 3), dtype=float)
    rot[..., 0, 0] = 1.0 - 2.0 * (y*y + z*z)
    rot[..., 0, 1] = 2.0 * (x*y - z*w)
    rot[..., 0, 2] = 2.0 * (x*z + y*w)
    rot[..., 1, 0] = 2.0 * (x*y + z*w)
    rot[..., 1, 1] = 1.0 - 2.0 * (x*x + z*z)
    rot[..., 1, 2] = 2.0 * (y*z - x*w)
    rot[..., 2, 0] = 2.0 * (x*z - y*w)
    rot[..., 2, 1] = 2.0 * (y*z + x*w)
    rot[..., 2, 2] = 1.0 - 2.0 * (x*x + y*y)
    return rot


def rot_to_euler_rad(rot) -> np.ndarray:
    rot = np.asarray(rot, dtype=float)

    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rot must have shape (..., 3, 3), got {rot.shape}")

    r00 = rot[..., 0, 0]
    r01 = rot[..., 0, 1]
    r02 = rot[..., 0, 2]
    r10 = rot[..., 1, 0]
    r11 = rot[..., 1, 1]
    r12 = rot[..., 1, 2]
    r22 = rot[..., 2, 2]

    y = np.arcsin(np.clip(r02, -1.0, 1.0))
    cos_y = np.cos(y)
    singular = np.abs(cos_y) < 1e-8

    x = np.where(
        singular,
        np.arctan2(r10, r11),
        np.arctan2(-r12, r22),
    )
    z = np.where(
        singular,
        0.0,
        np.arctan2(-r01, r00),
    )

    return np.stack([x, y, z], axis=-1)

def euler_rad_to_rot(euler: np.ndarray) -> np.ndarray:
    euler = np.asarray(euler, dtype=float)

    if euler.shape[-1] != 3:
        raise ValueError(f"euler must have shape (..., 3), got {euler.shape}")

    x, y, z = np.moveaxis(euler, -1, 0)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    rot = np.empty(euler.shape[:-1] + (3, 3), dtype=float)
    rot[..., 0, 0] = cy * cz
    rot[..., 0, 1] = -cy * sz
    rot[..., 0, 2] = sy
    rot[..., 1, 0] = sx * sy * cz + cx * sz
    rot[..., 1, 1] = -sx * sy * sz + cx * cz
    rot[..., 1, 2] = -sx * cy
    rot[..., 2, 0] = -cx * sy * cz + sx * sz
    rot[..., 2, 1] = cx * sy * sz + sx * cz
    rot[..., 2, 2] = cx * cy

    return rot


def euler_xyz_extrinsic_to_rot(euler: np.ndarray) -> np.ndarray:
    """Rotation matrix matching IsaacLab ``euler_xyz_from_quat`` output."""
    euler = np.asarray(euler, dtype=float)

    if euler.shape[-1] != 3:
        raise ValueError(f"euler must have shape (..., 3), got {euler.shape}")

    x, y, z = np.moveaxis(euler, -1, 0)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    rot = np.empty(euler.shape[:-1] + (3, 3), dtype=float)
    rot[..., 0, 0] = cz * cy
    rot[..., 0, 1] = cz * sy * sx - sz * cx
    rot[..., 0, 2] = cz * sy * cx + sz * sx
    rot[..., 1, 0] = sz * cy
    rot[..., 1, 1] = sz * sy * sx + cz * cx
    rot[..., 1, 2] = sz * sy * cx - cz * sx
    rot[..., 2, 0] = -sy
    rot[..., 2, 1] = cy * sx
    rot[..., 2, 2] = cy * cx

    return rot


def pos_quat_2_tf(pos, quat):
    rot = quat_to_rot(quat)
    tf = pos_rot_to_tf(pos, rot)
    return tf

def pose_2_tf(pos_wxyz):
    pos = pos_wxyz[..., :3]
    quat = pos_wxyz[..., 3:7]
    return pos_quat_2_tf(pos, quat)

def tf_2_keypoints(tf, cube_length = 0.3, out_type: str = "numpy"):
    pos, rot = tf_to_pos_rot(tf)
    
    axis_x = rot[..., 0]
    axis_y = rot[..., 1]
    axis_z = rot[..., 2]

    point_c = pos
    point_x = pos + cube_length * axis_x
    point_y = pos + cube_length * axis_y
    point_z = pos + cube_length * axis_z

    keypoints = np.concatenate([
        point_c, 
        point_x, 
        point_z,
        ], axis=-1)
    
    match out_type.lower():
        case "numpy":
            keypoints = np.asarray(keypoints, dtype=np.float32)
        case "torch":
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        case _:
            raise ValueError(f"out_type must be 'numpy' or 'torch', got {out_type}")
    
    return keypoints


# ======= 齐次变换矩阵 ======= #
def tf_reference(tf, ref_tf):
    """Express ``tf`` in the coordinate frame represented by ``ref_tf``."""

    match tf:
        case torch.Tensor():
            ref_tf_inv = torch.linalg.inv(ref_tf)
            return ref_tf_inv @ tf
        case np.ndarray():
            ref_tf_inv = np.linalg.inv(ref_tf)
            return ref_tf_inv @ tf
        case _:
            raise ValueError(f"tf must be np.ndarray or torch.Tensor, got {type(tf)}")
