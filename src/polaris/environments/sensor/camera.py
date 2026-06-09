import dataclasses
import numpy as np
import cv2

@dataclasses.dataclass
class GoPro_2_7K:

    fov = 155.0

    image_width = 2704
    image_height = 2028

    aspect_ratio = 1.0  # pixel x y ratio (square pixel)
    fx = 796.8545
    fy = fx * aspect_ratio

    # Assume perfectly centered
    cx = image_width / 2
    cy = image_height / 2

    # intrinsic matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0 ,1]])

    # Distortion
    # theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
    D = np.array([
            0.0,  # coef_a
            1/fx, # coef_b
            0.0,  # coef_c
            0.0,  # coef_d
            0.0,  # coef_e
            0.0,  # coef_f
        ])


# For UMI Trajectory Visualization
class FISHEYE_CAMERA_API:
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

    # 内参矩阵
    K = GoPro_2_7K.K

    # 鱼眼畸变
    POLYNOMIAL = GoPro_2_7K.D
    MAX_THETA  = GoPro_2_7K.fov / 2


    @classmethod
    def TF_ISC_TO_CAM(cls, tf_isc_relative: np.ndarray) -> np.ndarray:
        """Convert a relative IsaacSim transform into UMI/camera convention."""
        return (cls.TF_SHIFT.T @ tf_isc_relative @ cls.TF_SHIFT_INV_T).astype(np.float32)

    @classmethod
    def _invert_isaaclab_fisheye_polynomial(
        cls,
        theta: np.ndarray,
        fallback_focal: float,
    ) -> np.ndarray:
        
        a, b, c, d, e, f = cls.POLYNOMIAL

        if abs(float(b)) > 1e-8:
            radius = (theta - a) / b
        else:
            radius = fallback_focal * theta
        radius = np.maximum(radius, 0.0).astype(np.float32)

        for _ in range(8):
            radius2 = radius * radius
            radius3 = radius2 * radius
            radius4 = radius2 * radius2
            radius5 = radius4 * radius
            polynomial = a + b * radius + c * radius2 + d * radius3 + e * radius4 + f * radius5
            derivative = b + 2.0 * c * radius + 3.0 * d * radius2 + 4.0 * e * radius3 + 5.0 * f * radius4
            step = np.zeros_like(radius, dtype=np.float32)
            np.divide(polynomial - theta, derivative, out=step, where=np.abs(derivative) > 1e-8)
            radius = np.maximum(radius - step, 0.0)

        return radius.astype(np.float32)   


    @classmethod
    def PROJECT_CAM_LINEAR_XY(cls, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        z = points_cam[..., 2]

        # 只是用 z > 0 的点，因为有些点更新后会在镜头后面
        valid = np.isfinite(points_cam).all(axis=-1) & (z > 1e-6) 

        xy = np.full(points_cam.shape[:-1] + (2,), np.nan, dtype=np.float32)
        safe_z = np.where(valid, z, 1.0)
        xy[..., 0] = cls.K[0, 0] * points_cam[..., 0] / safe_z + cls.K[0, 2]
        xy[..., 1] = cls.K[1, 1] * points_cam[..., 1] / safe_z + cls.K[1, 2]

        xy[~valid] = np.nan
        valid &= np.isfinite(xy).all(axis=-1)

        return xy, valid


    @classmethod
    def PROJECT_CAM_FISHEYE_XY(cls, points_cam: np.ndarray) -> np.ndarray:
    
        z = points_cam[..., 2]

        # 只是用 z > 0 的点，因为有些点更新后会在镜头后面
        valid = np.isfinite(points_cam).all(axis=-1) & (z > 1e-6)

        xy = np.full(points_cam.shape[:-1] + (2,), np.nan, dtype=np.float32)
        safe_z = np.where(valid, z, 1.0)
        x = points_cam[..., 0] / safe_z
        y = points_cam[..., 1] / safe_z

        r_normalized = np.sqrt(x * x + y * y)
        theta = np.arctan(r_normalized)
        valid &= theta <= float(cls.MAX_THETA)


        radius = cls._invert_isaaclab_fisheye_polynomial(
            theta,
            fallback_focal=float(cls.K[0, 0]),
        )

        direction_scale = np.zeros_like(r_normalized, dtype=np.float32)
        np.divide(radius, r_normalized, out=direction_scale, where=r_normalized > 1e-8)

        xy[..., 0] = cls.K[0, 2] + x * direction_scale
        xy[..., 1] = cls.K[1, 2] + y * direction_scale

        xy[~valid] = np.nan
        valid &= np.isfinite(xy).all(axis=-1)

        return xy, valid
    

    @staticmethod
    def DRAW_PROJECTED_POINTS(
        img_debug: np.ndarray,
        xy: np.ndarray,
        valid: np.ndarray,
    ) -> None:

        if xy.ndim == 2:
            xy = xy[None, ...]
            valid = valid[None, ...]
        else:
            xy = xy.reshape((-1, xy.shape[-2], 2))
            valid = valid.reshape((-1, valid.shape[-1]))

        image_max = float(np.nanmax(img_debug)) if img_debug.size else 0.0
        normalized_float = np.issubdtype(img_debug.dtype, np.floating) and image_max <= 1.0
        color_scale   = 1.0 if normalized_float else 255.0
        start_color   = np.array([0.0, 220.0, 255.0], dtype=np.float32) * (color_scale / 255.0)
        end_color     = np.array([255.0, 60.0, 40.0], dtype=np.float32) * (color_scale / 255.0)
        outline_color = np.array([255.0, 255.0, 255.0], dtype=np.float32) * (color_scale / 255.0)

        def cv_color(color: np.ndarray) -> tuple:
            if img_debug.shape[-1] > 3:
                color = np.concatenate([color, np.array([color_scale], dtype=np.float32)])
            if np.issubdtype(img_debug.dtype, np.integer):
                return tuple(int(round(channel)) for channel in color)
            return tuple(float(channel) for channel in color)

        height, width = img_debug.shape[:2]
        for traj_xy, traj_valid in zip(xy, valid):
            if traj_xy.shape[0] == 0:
                continue
            denom = max(traj_xy.shape[0] - 1, 1)
            traj_valid = traj_valid & np.isfinite(traj_xy).all(axis=-1)
            points_i = np.zeros(traj_xy.shape, dtype=np.int32)
            points_i[traj_valid] = np.rint(traj_xy[traj_valid]).astype(np.int32)

            for idx in range(1, len(points_i)):
                if not (traj_valid[idx - 1] and traj_valid[idx]):
                    continue
                p0 = tuple(points_i[idx - 1])
                p1 = tuple(points_i[idx])
                color = start_color * (1.0 - idx / denom) + end_color * (idx / denom)
                cv2.line(img_debug, p0, p1, cv_color(outline_color), 4, lineType=cv2.LINE_AA)
                cv2.line(img_debug, p0, p1, cv_color(color), 2, lineType=cv2.LINE_AA)

            for idx, point in enumerate(points_i):
                if not traj_valid[idx]:
                    continue
                x, y = int(point[0]), int(point[1])
                if not (0 <= x < width and 0 <= y < height):
                    continue
                color = start_color * (1.0 - idx / denom) + end_color * (idx / denom)
                radius = 5 if idx in (0, len(points_i) - 1) else 3
                cv2.circle(img_debug, (x, y), radius + 1, cv_color(outline_color), -1, lineType=cv2.LINE_AA)
                cv2.circle(img_debug, (x, y), radius, cv_color(color), -1, lineType=cv2.LINE_AA)