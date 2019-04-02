"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""

from second.core import box_np_ops

def RotateIouSimilarity(boxes1, boxes2):
    """Class to compute similarity based on Intersection over Union (IOU) metric.
    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    return box_np_ops.riou_cc(boxes1, boxes2)

def NearestIouSimilarity(boxes1, boxes2):
    """Class to compute similarity based on the squared distance metric.

    This class computes pairwise similarity between two BoxLists based on the
    negative squared distance metric.
    """
    boxes1_bv = box_np_ops.rbbox2d_to_near_bbox(boxes1)
    boxes2_bv = box_np_ops.rbbox2d_to_near_bbox(boxes2)
    return box_np_ops.iou_jit(boxes1_bv, boxes2_bv, eps=0.0)

def DistanceSimilarity(boxes1, boxes2, distance_norm=2.0,
                       with_rotation=False, rotation_alpha=0.5):
    """Class to compute similarity based on Intersection over Area (IOA) metric.

    This class computes pairwise similarity between two BoxLists based on their
    pairwise intersections divided by the areas of second BoxLists.
    """
    return box_np_ops.distance_similarity(
        boxes1[..., [0, 1, -1]],
        boxes2[..., [0, 1, -1]],
        dist_norm=distance_norm,
        with_rotation=with_rotation,
        rot_alpha=rotation_alpha)
