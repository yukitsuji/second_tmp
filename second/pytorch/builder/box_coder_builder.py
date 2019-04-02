import numpy as np

from second.core.box_coders import GroundBox3dCoder
from second.pytorch.core import box_torch_ops


class GroundBox3dCoderTorch(GroundBox3dCoder):
    def encode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_encode(boxes, anchors, self.vec_encode,
                                               self.linear_dim)

    def decode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_decode(boxes, anchors, self.vec_encode,
                                               self.linear_dim)


def build(box_coder_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    cfg = box_coder_config.bev_box_coder
    return GroundBox3dCoderTorch(cfg.linear_dim, cfg.encode_angle_vector)
