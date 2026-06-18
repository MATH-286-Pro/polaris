import torch
import numpy as np
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from .. import tool_linalg


class TfFrameMarker:
    def __init__(
        self,
        prim_path: str,
        scale: tuple[float, float, float] = (0.01, 0.01, 0.01),
        debug=False,
    ) -> None:
        self.prim_path = prim_path
        self.scale = scale
        self.marker: VisualizationMarkers | None = None
        self.debug = debug

        if not self.debug:
            return

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = self.prim_path
        marker_cfg.markers["frame"].scale = self.scale
        self.marker = VisualizationMarkers(marker_cfg)

    def update(
        self,
        tf_isc_b: np.ndarray,
        device: torch.device,
    ) -> None:
        if not self.debug:
            return

        position, orientation = self._tf_to_marker_pose(tf_isc_b, device)
        self.marker.visualize(position, orientation)

    @staticmethod
    def _tf_to_marker_pose(tf: np.ndarray, device: torch.device):

        if tf.ndim > 2:
            tf = tf.reshape(-1, *tf.shape[-2:])[0]

        tf_tensor = torch.as_tensor(tf, dtype=torch.float32, device=device)
        positions = tf_tensor[:3, 3].unsqueeze(0)
        orientations = tool_linalg.quat_from_matrix(tf_tensor[:3, :3].unsqueeze(0))
        return positions, orientations