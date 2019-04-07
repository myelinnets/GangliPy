import torch


class PercentOfMaxBoosting(object):
    def __call__(self, input_tensor, boost_tensor, boost_percent):
        # type: (torch.Tensor, torch.Tensor, float) -> (torch.Tensor, torch.Tensor)
        max_val = torch.max(input_tensor.clone())
        max_val = torch.max(torch.zeros_like(max_val), max_val)
        boost_plus = max_val * boost_percent
        out_boost = boost_tensor + boost_plus
        return out_boost


class PercentClosenessBoosting(object):
    def __call__(self, input_tensor, boost_tensor, boost_percent):
        # type: (torch.Tensor, torch.Tensor, float) -> (torch.Tensor, torch.Tensor)
        max_val = torch.max(input_tensor.clone())
        max_val = torch.max(torch.zeros_like(max_val), max_val)
        min_val = torch.min(input_tensor.clone())
        dist = max_val - min_val
        percent_of_dist = 1.0 - ((max_val - input_tensor) / dist)
        boost_plus = percent_of_dist * boost_percent
        out_boost = boost_tensor + boost_plus
        return out_boost
