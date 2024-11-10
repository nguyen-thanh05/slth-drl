from torch.autograd import Function
import torch


def percentile(t, q):
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


class GetSubnet(Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k_val = percentile(scores, sparsity * 100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
    
    @staticmethod
    def backward(ctx, g):
        return g, None, None, None
