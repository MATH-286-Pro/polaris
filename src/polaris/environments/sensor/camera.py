import dataclasses
import numpy as np

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
    D_RAW = np.array([
        -6.629688912821408e-04,  # a
        1.2804984241915596e-03,  # b
        -2.2745668649115827e-07, # c
        8.128422232975493e-10,   # d
        -1.1063642471143477e-12, # e
        6.419046128976403e-16,   # f
    ])

    # Scale the fisheye image radius while preserving the relative distortion
    # shape: P_scaled(r) = P_raw(r / radius_scale).
    radius_scale = 1.104402594417232
    D = D_RAW / np.array(
        [
            1.0,
            radius_scale,
            radius_scale**2,
            radius_scale**3,
            radius_scale**4,
            radius_scale**5,
        ],
        dtype=np.float64,
    )

