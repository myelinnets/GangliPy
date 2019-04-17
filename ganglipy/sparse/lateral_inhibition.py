import torch


def apply_sparse_self_affector(
        tensor,
        sparse_tensor  # type: torch.Tensor
):
    """
    >>> t1 = torch.ones((5,5))
    >>> s1 = torch.sparse_coo_tensor(torch.LongTensor([[0],[0],[1],[1]]),torch.FloatTensor([-0.5]),(5,5,5,5))
    >>> apply_sparse_self_affector(t1, s1)
    tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 0.5000, 1.0000, 1.0000, 1.0000],
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


def apply_sparse_self_affector_const_val(
        tensor, adder,
        sparse_tensor  # type: torch.Tensor
):
    """
    >>> t1 = torch.ones((5,5))
    >>> s1 = torch.sparse_coo_tensor(torch.LongTensor([[0],[0],[1],[1]]),torch.FloatTensor([-0.5]),(5,5,5,5))
    >>> apply_sparse_self_affector(t1, s1)
    tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 0.5000, 1.0000, 1.0000, 1.0000],
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
    tens[list(coo_indices[t_rank:])] += torch.ones_like(tens[list(coo_indices[:t_rank])]) * adder
    return tens


def add_self_affector(inhibition_tensor, affector_index, affectee_index):
    """
        >>> s1 = torch.sparse_coo_tensor(torch.LongTensor([[0],[0],[1],[1]]),torch.FloatTensor([-0.5]),(5,5,5,5))
        >>> affector = torch.LongTensor([[2],[2]])
        >>> affectee = torch.LongTensor([[3],[3]])
        >>> add_self_affectors(s1, affector, affectee)
        tensor(indices=tensor([[0, 2],
                               [0, 2],
                               [1, 3],
                               [1, 3]]),
               values=tensor([ -0.5000, -0.1000]),
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
        >>> s1 = torch.sparse_coo_tensor(torch.LongTensor([[0],[0],[1],[1]]),torch.FloatTensor([-0.5]),(5,5,5,5))
        >>> affector = torch.LongTensor([[2,4],[2,3]])
        >>> affectee = torch.LongTensor([[3,2],[3,1]])
        >>> add_self_affectors(s1, affecto2r, affectee)
        tensor(indices=tensor([[0, 2, 4],
                               [0, 2, 3],
                               [1, 3, 2],
                               [1, 3, 1]]),
               values=tensor([-0.5000, -0.0100, -0.0100]),
               size=(5, 5, 5, 5), nnz=3, layout=torch.sparse_coo)


        :param tensor: Tensor to have self links applied to.
        :param sparse_tensor: Sparse tensor representing links of points in tensor to other points in tensor.
            Rank should be 2x tensor's rank.
        :return: tensor modified by self links.
        """
    for i in range(affector_indices.shape[1]):
        new_index = torch.cat((affector_indices[:, i:i + 1], affectee_indices[:, i:i + 1]), 0).type_as(
            inhibition_tensor._indices())
        if inhibition_tensor.is_cuda:
            new_index = new_index.cuda()
        if inhibition_tensor._indices().shape[1] != 0:

            indices_equality = (
                    inhibition_tensor._indices() == new_index.repeat(1, inhibition_tensor._indices().shape[1]))
            indices_dim = inhibition_tensor._indices().shape[0]
            column_check = torch.ones((1, indices_dim))
            column_sum = torch.mm(column_check, indices_equality.type_as(column_check))
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

#  todo: investigate growing and moving self affectors instead of adding
