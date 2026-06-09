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
    D = np.array([
            0.0,  # coef_a
            1/fx, # coef_b
            0.0,  # coef_c
            0.0,  # coef_d
            0.0,  # coef_e
            0.0,  # coef_f
        ])