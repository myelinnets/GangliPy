import torch
from torch import autograd
from torch import nn
import math


# boost -> lateral-inhibit -> k-winners
# boost (update boosting) <- k-winners (winners)
# lateral-inhibit (store losers, update chokes) <- k-winners (winners)

# todo: create KWinnersBoose _Function_, not module, and implement backprop in it.
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
        if boost is not None:
            boosted = tensor + boost
        else:
            boosted = tensor
        boosted = boosted.clone()
        _, argsort = boosted.sort(dim=1, descending=True)
        k_active = math.ceil(sparsity * embedding_size)
        active_indices = argsort[:, :k_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        mask_active[torch.arange(batch_size).unsqueeze(1), active_indices] = 1
        tensor[~mask_active] = 0
        ctx.save_for_backward(tensor)
        return tensor


class _KWinnersBoostFunc(autograd.Function):
    # adapted from: https://discuss.pytorch.org/t/k-winner-take-all-advanced-indexing/24348

    @staticmethod
    def backward(ctx, *grad_outputs):
        result, boost_tensor, boost_percent = ctx.saved_tensors
        backprop = grad_outputs[0] * result
        return backprop, None, None, None, None

    @staticmethod
    def forward(ctx, tensor, sparsity, boost_tensor, boost_percent,
                boost_method):  # type: (Any, torch.Tensor, float) -> torch.Tensor
        batch_size, embedding_size = tensor.shape[:2]
        boost_tensor = boost_method(tensor, boost_tensor, boost_percent)
        tensor_above = torch.where(tensor > 0, tensor, torch.zeros_like(tensor))
        boosted = tensor_above + boost_tensor
        boosted = boosted.clone()
        test, argsort = boosted.sort(dim=1, descending=True)
        k_active = math.ceil(sparsity * embedding_size)
        active_indices = argsort[:, :k_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        mask_active[torch.arange(batch_size).unsqueeze(1), active_indices] = 1
        tensor[~mask_active] = 0
        tensor = torch.where(tensor > 0, torch.ones_like(tensor), torch.zeros_like(tensor))
        if torch.sum(tensor).item() == 0:
            tensor[mask_active] = 1
        ctx.save_for_backward(tensor, boost_tensor, boost_percent)
        boost_tensor[mask_active] = 0

        return tensor, boost_tensor


class PercentOfMaxBoosting(object):
    def __call__(self, input_tensor, boost_tensor, boost_percent):
        # type: (torch.Tensor, torch.Tensor, float) -> (torch.Tensor, torch.Tensor)
        max_val = torch.max(input_tensor.clone())
        max_val = torch.max(torch.zeros_like(max_val), max_val)
        boost_plus = max_val * boost_percent
        out_boost = boost_tensor + boost_plus
        return out_boost


class KWinnersBoost(nn.Module):
    def __init__(self, boost_method=PercentOfMaxBoosting()):
        super().__init__()
        self.boost_method = boost_method
        self.func = _KWinnersBoostFunc()
        self.func_apply = self.func.apply
        self.register_parameter('boost_tensor', None)

    def forward(self, tensor, sparsity=0.02, boost_percent=torch.tensor(1e-8)):
        if boost_percent is None:
            raise ValueError("boost_percent cannot be None.")
        if self.boost_tensor is None:
            self.boost_tensor = nn.Parameter(torch.zeros_like(tensor))

        tensor, boost_tensor = self.func_apply(tensor, sparsity, self.boost_tensor, boost_percent, self.boost_method)
        self.boost_tensor = nn.Parameter(boost_tensor)
        return tensor
