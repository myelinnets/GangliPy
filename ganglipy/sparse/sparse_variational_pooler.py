# boost -> lateral-inhibit -> k-winners
# boost (update boosting) <- k-winners (winners)
# lateral-inhibit (store losers, update chokes) <- k-winners (winners)

import torch
from torch import nn
from ganglipy.sparse.boosting import PercentClosenessBoosting, boost, boost_to_min_sparsity
from ganglipy.sparse.lateral_inhibition import apply_sparse_self_affector_const_val, add_self_affectors
from ganglipy.sparse.kwinners_boosted import _KWinnersBoostFunc
from ganglipy.sparse.kwinners import k_winners_positive


class _KWinnersInhibitFunc(_KWinnersBoostFunc):
    # adapted from: https://discuss.pytorch.org/t/k-winner-take-all-advanced-indexing/24348

    @staticmethod
    def backward(ctx, *grad_outputs):
        result, = ctx.saved_tensors
        backprop = grad_outputs[0] * result
        return backprop, None, None, None, None

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx,
                tensor,
                sparsity,
                boosting,
                inhibition_tensor
                ):
        boost_tensor, boosted = boost(tensor, *boosting)
        inhibited = apply_sparse_self_affector_const_val(boosted, 0.01, inhibition_tensor)
        tensor, rankings = k_winners_positive(inhibited, sparsity)
        ctx.save_for_backward(tensor)  # must not include pure boost activations
        tensor = boost_to_min_sparsity(tensor, boost_tensor, sparsity)
        top_active = torch.zeros((len(tensor.shape), 50))
        top_active[1, :] = rankings[0, 0:1, 0, 0]
        max_sparsity = sparsity[1]

        batch_size, embedding_size = tensor.shape[:2]
        max_active = int(torch.ceil(max_sparsity * embedding_size).item())
        other_active = torch.zeros((len(tensor.shape), 50))
        other_active[1, :] = rankings[0, max(rankings.shape[1] - 150, 1):max(rankings.shape[1] - 100, 1), 0, 0]
        inhibition_tensor = add_self_affectors(inhibition_tensor, top_active, other_active)

        # todo: move this out to its own function
        desired_max_total_connections = 1000 * 1000 * tensor.shape[1]
        conn = inhibition_tensor._values().shape[0]
        subtraction = (conn ** 2) / (conn ** 2 + desired_max_total_connections * 2)
        subtraction *= torch.max(torch.abs(inhibition_tensor._values()))
        new_vals = torch.min((inhibition_tensor._values() + torch.ones_like(inhibition_tensor._values()) * subtraction),
                             torch.zeros_like(inhibition_tensor._values()))
        new_vals = torch.max(new_vals, torch.ones_like(new_vals) * -1.0)
        new_nonzeros = new_vals.nonzero()
        new_vals = new_vals[new_nonzeros[:, 0]]
        new_indices = inhibition_tensor._indices()[:, new_nonzeros[:, 0]]
        inhibition_tensor = torch.sparse_coo_tensor(new_indices, new_vals, inhibition_tensor.shape)

        boost_tensor = torch.where(tensor > 0, torch.zeros_like(boost_tensor), boost_tensor)

        return tensor, boost_tensor, inhibition_tensor


class SparseVariationalPooler(nn.Module):
    def __init__(self, boost_method=PercentClosenessBoosting()):
        super().__init__()
        self.boost_method = boost_method
        self.func = _KWinnersInhibitFunc()
        self.func_apply = self.func.apply
        self.register_parameter('boost_tensor', None)
        self.register_parameter('inhibition_tensor', None)

    def forward(self, tensor, sparsity=torch.Tensor([0.002, 0.02]), boost_percent=torch.tensor(1e-9)):
        if boost_percent is None:
            raise ValueError("boost_percent cannot be None.")
        if self.boost_tensor is None:
            self.boost_tensor = nn.Parameter(torch.zeros_like(tensor))
        if self.inhibition_tensor is None:
            self.inhibition_tensor = nn.Parameter(
                torch.sparse_coo_tensor([[]] * len(tensor.shape) * 2, [], list(tensor.shape) * 2).cuda())

        tensor, boost_tensor, inhibition_tensor = self.func_apply(tensor,
                                                                  sparsity,
                                                                  (self.boost_tensor, boost_percent, self.boost_method),
                                                                  self.inhibition_tensor)
        self.boost_tensor = nn.Parameter(boost_tensor)
        self.inhibition_tensor = nn.Parameter(inhibition_tensor)
        return tensor
