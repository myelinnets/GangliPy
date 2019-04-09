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
    if sparse_tensor._values().shape[0] == 0:
        return tensor
    coo_indices = sparse_tensor._indices()
    tens = tensor.clone()
    tens[list(coo_indices[t_rank:])] += tens[list(coo_indices[:t_rank])] * sparse_tensor._values()
    return tens


def add_self_affector(inhibition_tensor, affector_index, affectee_index):
    """
        >>> s1 = torch.sparse_coo_tensor(torch.LongTensor([[0],[0],[1],[1]]),torch.FloatTensor([0.5]),(5,5,5,5))
        >>> affector = torch.LongTensor([[2],[2]])
        >>> affectee = torch.LongTensor([[3],[3]])
        >>> add_self_affectors(s1, affector, affectee)
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

def add_self_affectors(inhibition_tensor, affector_indices, affectee_indices, affect_value=-0.01):
    """
        >>> s1 = torch.sparse_coo_tensor(torch.LongTensor([[0],[0],[1],[1]]),torch.FloatTensor([0.5]),(5,5,5,5))
        >>> affector = torch.LongTensor([[2,4],[2,3]])
        >>> affectee = torch.LongTensor([[3,2],[3,1]])
        >>> add_self_affectors(s1, affector, affectee)
        tensor(indices=tensor([[0, 2, 4],
                               [0, 2, 3],
                               [1, 3, 2],
                               [1, 3, 1]]),
               values=tensor([ 0.5000, -0.0100, -0.0100]),
               size=(5, 5, 5, 5), nnz=3, layout=torch.sparse_coo)


        :param tensor: Tensor to have self links applied to.
        :param sparse_tensor: Sparse tensor representing links of points in tensor to other points in tensor.
            Rank should be 2x tensor's rank.
        :return: tensor modified by self links.
        """
    for i in range(affector_indices.shape[1]):
        new_index = torch.cat((affector_indices[:,i:i+1], affectee_indices[:,i:i+1]), 0).type_as(inhibition_tensor._indices())
        if inhibition_tensor.is_cuda:
            new_index = new_index.cuda()
        if inhibition_tensor._indices().shape[1]!=0:

            indices_equality = (inhibition_tensor._indices() == new_index.repeat(1, inhibition_tensor._indices().shape[1]))
            indices_dim = inhibition_tensor._indices().shape[0]
            column_check = torch.ones((1,indices_dim))
            column_sum = torch.mm(column_check,indices_equality.type_as(column_check))
            found_index = (column_sum == indices_dim).nonzero()
        else:
            found_index = []

        sparse_indices = inhibition_tensor._indices()
        sparse_values = inhibition_tensor._values()
        sparse_shape = inhibition_tensor.shape

        if len(found_index) == 1:
            found_index = found_index[0]
            sparse_values[found_index] += affect_value
        elif len(found_index) == 0:
            sparse_indices = torch.cat((sparse_indices, new_index), 1)
            sparse_values = torch.cat((sparse_values, torch.Tensor([affect_value]).type_as(sparse_values)), 0)
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
        new_shape = list(tensor.shape)
        new_shape[1]=max_active
        t_select = tensor[:, active_indices].view(new_shape).nonzero()
        argsort = argsort[:, t_select[:,1]]
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
        tensor, rankings = _KWinnersBoostFunc.run_k_winners_positive(inhibited, sparsity)
        ctx.save_for_backward(tensor)  # must not include pure boost activations
        tensor = _KWinnersBoostFunc.choose_boosted_to_satisfy_minimum(tensor, boost_tensor, sparsity)
        top_active = torch.zeros((len(tensor.shape), 20))
        top_active[1,:] = rankings[0,0:1,0,0]
        max_sparsity = sparsity[1]

        batch_size, embedding_size = tensor.shape[:2]
        max_active = int(torch.ceil(max_sparsity * embedding_size).item())
        other_active = torch.zeros((len(tensor.shape), 20))
        other_active[1, :] = rankings[0,max(rankings.shape[1]-20,1):max_active,0,0]
        inhibition_tensor = add_self_affectors(inhibition_tensor, top_active, other_active)

        # todo: move this out to its own function
        desired_max_total_connections = 1000*1000*1000*1000*1000*tensor.shape[1]
        conn = inhibition_tensor._values().shape[0]
        subtraction = (conn**2) / (conn**2 + desired_max_total_connections*2)
        subtraction *= torch.max(torch.abs(inhibition_tensor._values()))
        new_vals = torch.min((inhibition_tensor._values() + torch.ones_like(inhibition_tensor._values())*subtraction), torch.zeros_like(inhibition_tensor._values()))
        new_vals = torch.max(new_vals, torch.ones_like(new_vals)*-1.0)
        new_nonzeros = new_vals.nonzero()
        new_vals = new_vals[new_nonzeros[:,0]]
        new_indices = inhibition_tensor._indices()[:,new_nonzeros[:,0]]
        inhibition_tensor = torch.sparse_coo_tensor(new_indices, new_vals, inhibition_tensor.shape)


        boost_tensor = torch.where(tensor > 0, torch.zeros_like(boost_tensor), boost_tensor)

        return tensor, boost_tensor, inhibition_tensor


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

        tensor, boost_tensor, inhibition_tensor = self.func_apply(tensor,
                                               sparsity,
                                               (self.boost_tensor, boost_percent, self.boost_method),
                                               self.inhibition_tensor)
        self.boost_tensor = nn.Parameter(boost_tensor)
        self.inhibition_tensor = nn.Parameter(inhibition_tensor)
        return tensor
