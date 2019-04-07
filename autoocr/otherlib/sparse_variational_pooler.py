# boost -> lateral-inhibit -> k-winners
# boost (update boosting) <- k-winners (winners)
# lateral-inhibit (store losers, update chokes) <- k-winners (winners)

import torch
from torch import autograd
import math
from torch import nn
from autoocr.otherlib.boosting import PercentClosenessBoosting
from torch import sparse


def apply_sparse_self_affector(
        tensor,
        sparse_tensor  # type: torch.Tensor
):
    """
    >>> t1 = torch.ones((5,5))
    >>> s1 = torch.sparse_coo_tensor(torch.LongTensor([[0],[0],[1],[1]]),torch.FloatTensor([0.5]),(5,5,5,5))
    >>> apply_sparse_self_affector(t1, s1)
    tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.5000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])

    :param tensor: Tensor to have self links applied to.
    :param sparse_tensor: Sparse tensor representing links of points in tensor to other points in tensor.
        Rank should be 2x tensor's rank.
    :return: tensor modified by self links.
    """
    t_rank = int(len(tensor.shape))
    coo_indices = sparse_tensor._indices()
    tensor[list(coo_indices[t_rank:])] += tensor[list(coo_indices[:t_rank])] * sparse_tensor._values()
    return tensor


def add_self_affector(inhibition_tensor, affector_index, affectee_index):
    """
        >>> s1 = torch.sparse_coo_tensor(torch.LongTensor([[0],[0],[1],[1]]),torch.FloatTensor([0.5]),(5,5,5,5))
        >>> affector = torch.LongTensor([[2],[2]])
        >>> affectee = torch.LongTensor([[3],[3]])
        >>> add_self_affector(s1, affector, affectee)
        tensor(indices=tensor([[0, 2],
                               [0, 2],
                               [1, 3],
                               [1, 3]]),
               values=tensor([ 0.5000, -0.1000]),
               size=(5, 5, 5, 5), nnz=2, layout=torch.sparse_coo)


        :param tensor: Tensor to have self links applied to.
        :param sparse_tensor: Sparse tensor representing links of points in tensor to other points in tensor.
            Rank should be 2x tensor's rank.
        :return: tensor modified by self links.
        """
    new_index = torch.cat((affector_index, affectee_index), 0)
    found_index = (inhibition_tensor._indices() == new_index).nonzero()
    sparse_indices = inhibition_tensor._indices()
    sparse_values = inhibition_tensor._values()
    sparse_shape = inhibition_tensor.shape

    if len(found_index) == 1:
        found_index = found_index[0]
        sparse_values[found_index] -= 0.01
    elif len(found_index) == 0:
        sparse_indices = torch.cat((sparse_indices, new_index), 1)
        sparse_values = torch.cat((sparse_values, torch.Tensor([-0.01]).type_as(sparse_values)), 0)
    else:
        raise RuntimeError("Sparse tensor has duplicate entries.")

    inhibition_tensor = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_shape)
    return inhibition_tensor


class _KWinnersBoostFunc(autograd.Function):
    # adapted from: https://discuss.pytorch.org/t/k-winner-take-all-advanced-indexing/24348

    @staticmethod
    def backward(ctx, *grad_outputs):
        result, = ctx.saved_tensors
        backprop = grad_outputs[0] * result
        return backprop, None, None, None, None

    @staticmethod
    def run_boosting(tensor, boosting):
        boost_tensor = boosting[0]
        boost_percent = boosting[1]
        boost_method = boosting[2]
        boost_tensor = boost_method(tensor, boost_tensor, boost_percent)
        tensor_above = torch.where(tensor > 0, tensor, torch.zeros_like(tensor))
        boosted = tensor_above + boost_tensor
        boosted = boosted.clone()
        return boost_tensor, boosted

    @staticmethod
    def run_k_winners_positive(tensor, sparsity):
        batch_size, embedding_size = tensor.shape[:2]

        test, argsort = tensor.sort(dim=1, descending=True)
        max_sparsity = sparsity[1]
        max_active = int(torch.ceil(max_sparsity * embedding_size).item())

        active_indices = argsort[:, :max_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        mask_active[torch.arange(batch_size).unsqueeze(1), active_indices] = 1
        tensor[~mask_active] = 0
        tensor = torch.where(tensor > 0, torch.ones_like(tensor), torch.zeros_like(tensor))
        return tensor, argsort

    @staticmethod
    def choose_boosted_to_satisfy_minimum(tensor, boost_tensor, sparsity):
        batch_size, embedding_size = tensor.shape[:2]

        min_sparsity = sparsity[0]
        embedding_size = tensor.shape[1]
        min_active = int(torch.floor(min_sparsity * embedding_size).item())

        actually_active = torch.sum(tensor).item()
        if actually_active < min_active:
            test, boostsort = boost_tensor.sort(dim=1, descending=True)
            to_activate = math.ceil(min_active - actually_active)
            boost_indices = boostsort[:, :to_activate]
            mask_boost = torch.ByteTensor(tensor.shape).zero_()
            mask_boost[torch.arange(batch_size).unsqueeze(1), boost_indices] = 1
            tensor[mask_boost] = 1
        return tensor

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx,
                tensor,
                sparsity,
                boosting,
                inhibition_tensor
                ):
        boost_tensor, boosted = _KWinnersBoostFunc.run_boosting(tensor, boosting)
        inhibited = apply_sparse_self_affector(boosted, inhibition_tensor)
        tensor, rankings = _KWinnersBoostFunc.run_k_winners_positive(boosted, sparsity)
        ctx.save_for_backward(tensor)  # must not include pure boost activations
        tensor = _KWinnersBoostFunc.choose_boosted_to_satisfy_minimum(tensor, boost_tensor, sparsity)
        if len(rankings) > 1:
            top_active = tensor[rankings[0]]
            top_active = top_active.repeat(1, len(rankings.shape) - 1)
            other_active = tensor[rankings[1:]]
            for t, o in zip(top_active, other_active):
                inhibition_tensor = add_self_affector(inhibition_tensor, t, o)

        # todo: randomly decrease other dendrites depending on sparse coo density and algo here

        boost_tensor = torch.where(tensor > 0, torch.zeros_like(boost_tensor), boost_tensor)

        return tensor, boost_tensor


class SparseVariationalPooler(nn.Module):
    def __init__(self, boost_method=PercentClosenessBoosting()):
        super().__init__()
        self.boost_method = boost_method
        self.func = _KWinnersBoostFunc()
        self.func_apply = self.func.apply
        self.register_parameter('boost_tensor', None)
        self.register_parameter('inhibition_tensor', None)

    def forward(self, tensor, sparsity=torch.Tensor([0.002, 0.02]), boost_percent=torch.tensor(1e-8)):
        if boost_percent is None:
            raise ValueError("boost_percent cannot be None.")
        if self.boost_tensor is None:
            self.boost_tensor = nn.Parameter(torch.zeros_like(tensor))
        if self.inhibition_tensor is None:
            self.inhibition_tensor = nn.Parameter(torch.sparse_coo_tensor([[]]*len(tensor.shape)*2, [], list(tensor.shape) * 2).cuda())

        tensor, boost_tensor = self.func_apply(tensor,
                                               sparsity,
                                               (self.boost_tensor, boost_percent, self.boost_method),
                                               self.inhibition_tensor)
        self.boost_tensor = nn.Parameter(boost_tensor)
        return tensor
