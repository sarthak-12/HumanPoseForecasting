import torch
from torch import Tensor


def euclidean_distance(x: Tensor, y: Tensor, dim) -> Tensor:
    """Returns the Euclidean Distance between two tensors of the same shape"""
    return torch.sqrt(torch.sum((x - y)**2, dim))


def MPJPE(predictions: Tensor, ground_truth: Tensor, heatmap: bool = False) -> Tensor:
    """Returns the Mean Per Joint Position Error for a sequence of predicted frames"""
    if heatmap:
        h, w = predictions.shape[-2:]
        predictions = predictions.reshape(-1, h, w)
        ground_truth = ground_truth.reshape(-1, h, w)
        dim = (-1, -2)
    else:
        predictions = predictions.reshape(-1, 2)
        ground_truth = ground_truth.reshape(-1, 2)
        dim = (-1,)
    errors = euclidean_distance(predictions, ground_truth, dim)
    return torch.mean(errors, dtype=torch.float)


def PDJ(predictions: Tensor, ground_truth: Tensor, heatmap: bool = False) -> Tensor:
    """Returns the Percantage of Predicted Joints for a sequence of predicted frames"""
    if heatmap:
        h, w = predictions.shape[-2:]
        predictions = predictions.reshape(-1, 17, h, w)
        ground_truth = ground_truth.reshape(-1, 17, h, w)
        dim = (-1, -2)
    else:
        predictions = predictions.reshape(-1, 17, 2)
        ground_truth = ground_truth.reshape(-1, 17, 2)
        dim = (-1,)
    # calculating the torso diameter as the distance between the pelvis (1st joint) and the shoulder center (9th joint)
    pelvis = ground_truth[:, 0]
    shoulder_center = ground_truth[:, 8]
    # threshold for each image is torso diameter * 0.2
    thresholds = euclidean_distance(pelvis, shoulder_center, dim) * 0.2
    errors = euclidean_distance(predictions, ground_truth, dim)
    detected_joints = 0
    for i, frame_errors in enumerate(errors):
        detected_joints += torch.sum(frame_errors < thresholds[i])
    percentage = detected_joints / (len(errors) * 17)
    return percentage
