import torch.nn as nn
import torch


class DiscountedMSELoss(nn.Module):
    """
    Loss equivalent to nn.MSELoss with the following modification: 
    mean along the temporal dimension is replaced by a discounted mean.
    Specifically for [x_0, x1, ..., x_n] the discounted mean is 
    (x_0 + p*x_1 + ... + p^n*x_n)/(1 + p + ... + p^n) given discount factor p. 
    """
    def __init__(self, discount=0.9):
        """
        Args:
            discount (float): discount factor
        """
        super(DiscountedMSELoss, self).__init__()

        self.discount = discount

    def forward(self, src, tgt):
        """
        Compute discounted MSE loss.

        Args: 
            src (tensor): source data with shape [B, T, ...]

            tgt (tensor): target data with shape [B, T, ...]

        Returns: scalar loss.
        """
        B, T = src.shape[0], src.shape[1]
        loss = torch.square(src-tgt).view(B, T, -1)
        loss = torch.mean(loss, [0, 2,])
        coef = torch.pow(torch.full([T], self.discount, device=src.device), torch.arange(T, device=src.device))
        return torch.sum(loss*coef)/torch.sum(coef)
    

class DiscountedL1Loss(nn.Module):
    """
    Loss equivalent to nn.L1Loss with the following modification: 
    mean along the temporal dimension is replaced by a discounted mean.
    Specifically for [x_0, x1, ..., x_n] the discounted mean is 
    (x_0 + p*x_1 + ... + p^n*x_n)/(1 + p + ... + p^n) given discount factor p. 
    """
    def __init__(self, discount=0.9):
        """
        Args:
            discount (float): discount factor
        """
        super(DiscountedL1Loss, self).__init__()

        self.discount = discount

    def forward(self, src, tgt):
        """
        Compute discounted L1 loss.

        Args: 
            src (tensor): source data with shape [B, T, ...]

            tgt (tensor): target data with shape [B, T, ...]

        Returns: scalar loss.
        """
        B, T = src.shape[0], src.shape[1]
        loss = torch.abs(src-tgt).view(B, T, -1)
        loss = torch.mean(loss, [0, 2,])
        coef = torch.pow(torch.full([T], self.discount, device=src.device), torch.arange(T, device=src.device))
        return torch.sum(loss*coef)/torch.sum(coef)