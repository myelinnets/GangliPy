import torch
from torch import autograd
from torch import nn
import math


class KWinnersTakeAll(autograd.Function):
    # adapted from: https://discuss.pytorch.org/t/k-winner-take-all-advanced-indexing/24348

    @staticmethod
    def backward(ctx, *grad_outputs):
        result, = ctx.saved_tensors
        backprop = grad_outputs[0] * result
        # print(backprop)
        return backprop, None, None

    @staticmethod
    def forward(ctx, tensor, sparsity):  # type: (Any, torch.Tensor, float) -> torch.Tensor
        batch_size, embedding_size = tensor.shape[:2]
        _, argsort = tensor.sort(dim=1, descending=True)
        k_active = math.ceil(sparsity * embedding_size)
        active_indices = argsort[:, :k_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        mask_active[torch.arange(batch_size).unsqueeze(1), active_indices] = 1
        tensor[~mask_active] = 0
        ctx.save_for_backward(tensor)
        return tensor
