import torch
from torch import autograd
from torch import nn
from ganglipy.sparse.boosting import PercentClosenessBoosting, boost, boost_to_min_sparsity
from ganglipy.sparse.kwinners import k_winners_positive


class _KWinnersBoostFunc(autograd.Function):
    # adapted from: https://discuss.pytorch.org/t/k-winner-take-all-advanced-indexing/24348

    @staticmethod
    def backward(ctx, *grad_outputs):
        result, = ctx.saved_tensors
        backprop = grad_outputs[0] * result
        return backprop, None, None, None, None

    @staticmethod
    def forward(ctx, tensor, sparsity, boosting):  # type: (Any, torch.Tensor, float) -> torch.Tensor
        boost_tensor, boosted = boost(tensor, *boosting)
        tensor, rankings = k_winners_positive(boosted, sparsity)
        ctx.save_for_backward(tensor)  # must not include pure boost activations
        tensor = boost_to_min_sparsity(tensor, boost_tensor, sparsity)
        boost_tensor = torch.where(tensor > 0, torch.zeros_like(boost_tensor), boost_tensor)
        return tensor, boost_tensor


class KWinnersBoost(nn.Module):
    def __init__(self, boost_method=PercentClosenessBoosting()):
        super().__init__()
        self.boost_method = boost_method
        self.func = _KWinnersBoostFunc()
        self.func_apply = self.func.apply
        self.register_parameter('boost_tensor', None)

    def forward(self, tensor, sparsity=torch.Tensor([0.002, 0.02]), boost_percent=torch.tensor(1e-8)):
        if boost_percent is None:
            raise ValueError("boost_percent cannot be None.")
        if self.boost_tensor is None:
            self.boost_tensor = nn.Parameter(torch.zeros_like(tensor))

        tensor, boost_tensor = self.func_apply(tensor, sparsity, self.boost_tensor, boost_percent, self.boost_method)
        self.boost_tensor = nn.Parameter(boost_tensor)
        return tensor
