import torch
from torch import autograd
import math

if False:
    from typing import Any


def k_winners_positive(tensor, sparsity):
    batch_size, embedding_size = tensor.shape[:2]

    test, argsort = tensor.sort(dim=1, descending=True)
    max_sparsity = sparsity[1]
    max_active = int(torch.ceil(max_sparsity * embedding_size).item())

    active_indices = argsort[:, :max_active]
    mask_active = torch.ByteTensor(tensor.shape).zero_()
    mask_active[torch.arange(batch_size).unsqueeze(1), active_indices] = 1
    tensor[~mask_active] = 0
    tensor = torch.where(tensor > 0, torch.ones_like(tensor), torch.zeros_like(tensor))
    new_shape = list(tensor.shape)
    new_shape[1] = max_active
    t_select = tensor[:, active_indices].view(new_shape).nonzero()
    argsort = argsort[:, t_select[:, 1]]
    return tensor, argsort


# noinspection PyMethodOverriding
class KWinnersTakeAll(autograd.Function):
    """
    A simple k-winners take all function, with backpropogation. Requires boosting to work well compared to any dense
      neural netowkrs.


    """

    # adapted from: https://discuss.pytorch.org/t/k-winner-take-all-advanced-indexing/24348

    @staticmethod
    def backward(ctx, *grad_outputs):
        result, = ctx.saved_tensors
        backprop = grad_outputs[0] * result
        return backprop, None, None

    @staticmethod
    def forward(ctx, tensor, sparsity):  # type: (Any, torch.Tensor, float) -> torch.Tensor
        """ K-winners forward pass.

        :param ctx: Context of the neural network using this function.
        :param tensor: Input dense tensor.
        :param sparsity: float value between 0 and 1 representing percenta sparsity.
        :return: the tensor with only (sparsity*100)% of neurons activated.
        """
        tensor, _ = k_winners_positive(tensor, sparsity)
        ctx.save_for_backward(tensor)
        return tensor
