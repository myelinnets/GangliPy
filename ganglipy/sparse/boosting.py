import torch
import math

if False:
    from numbers import Real


class PercentOfMaxBoosting(object):
    def __call__(self, input_tensor, boost_tensor, boost_percent):
        # type: (torch.Tensor, torch.Tensor, Real)->torch.Tensor
        """ Sets the boosting increment as a percent of the difference between max neuron activation and 0.

        :param input_tensor: pre-boosted input.
        :param boost_tensor: boost-energy values per neuron in input.
        :param boost_percent: percent of how boosting should influence firing compared to just input.
        :return:
        """
        max_val = torch.max(input_tensor.clone())
        max_val = torch.max(torch.zeros_like(max_val), max_val)
        boost_plus = max_val * boost_percent
        out_boost = boost_tensor + boost_plus
        return out_boost


class PercentClosenessBoosting(object):
    def __call__(self, input_tensor, boost_tensor, boost_percent):
        # type: (torch.Tensor, torch.Tensor, Real)->torch.Tensor
        """ Sets the boosting increment as a percent of the difference between max/min neuron activation.

        :param input_tensor: pre-boosted input.
        :param boost_tensor: boost-energy values per neuron in input.
        :param boost_percent: percent of how boosting should influence firing compared to just input.
        :return:
        """
        max_val = torch.max(input_tensor.clone())
        max_val = torch.max(torch.zeros_like(max_val), max_val)
        min_val = torch.min(input_tensor.clone())
        dist = max_val - min_val
        percent_of_dist = 1.0 - ((max_val - input_tensor) / dist)
        boost_plus = percent_of_dist * boost_percent
        out_boost = boost_tensor + boost_plus
        return out_boost


def boost(tensor, boost_tensor, boost_percent, boost_method):
    boost_tensor = boost_method(tensor, boost_tensor, boost_percent)
    tensor_above = torch.where(tensor > 0, tensor, torch.zeros_like(tensor))
    boosted = tensor_above + boost_tensor
    boosted = boosted.clone()
    return boost_tensor, boosted


def boost_to_min_sparsity(tensor, boost_tensor, sparsity):
    batch_size, embedding_size = tensor.shape[:2]

    min_sparsity = sparsity[0]
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
