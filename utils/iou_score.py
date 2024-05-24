import torch
from torch import Tensor

def pixel_wise_iou(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    intersection = (input * target).sum(dim=sum_dim)
    union = (input + target).sum(dim=sum_dim) - intersection

    iou = (intersection + epsilon) / (union + epsilon)
    print(iou.mean())
    return iou.mean()


def multiclass_pixel_wise_iou(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return pixel_wise_iou(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def pixel_wise_iou_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    fn = multiclass_pixel_wise_iou if multiclass else pixel_wise_iou
    return 1 - fn(input, target, reduce_batch_first=True)
