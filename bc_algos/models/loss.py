import torch.nn as nn
import torch


class DiscountedMSELoss(nn.Module):
    """
    Loss equivalent to nn.MSELoss with the following modifications: 

    1) mean along the temporal dimension is replaced by a discounted mean.
        The discounted mean for [x_0, x_1, ..., x_n] given discount factor p is 
        (x_0 + p*x_1 + ... + p^n*x_n)/(1 + p + ... + p^n). 

    2) option to mask loss
    """
    def __init__(self, discount=1.):
        """
        Args:
            discount (float): discount factor
        """
        super(DiscountedMSELoss, self).__init__()

        assert isinstance(discount, float), "discount factor must be a float"

        self.discount = discount

    @classmethod
    def factory(cls, config):
        """
        Create a DiscountedMSELoss instance from config.

        Args:
            config (addict): config object

        Returns: DiscountedMSELoss instance.
        """
        return cls(discount=config.train.discount)

    def forward(self, src, tgt, mask=None):
        """
        Compute discounted MSE loss.

        Args: 
            src (tensor): source data of shape [B, T, ...]

            tgt (tensor): target data of shape [B, T, ...]

            mask (tensor): (optional) binary mask of shape [B, T]

        Returns: scalar loss.
        """
        B, T = src.shape[0], src.shape[1]
        loss = torch.square(tgt-src).view(B, T, -1)
        loss = torch.mean(loss, -1)
        coef = torch.pow(torch.full([T], self.discount).to(src.device), torch.arange(T).to(src.device))
        if mask is not None:
            coef = mask * coef
        return torch.mean(torch.sum(coef*loss, dim=-1)/torch.sum(coef, dim=-1))


class DiscountedL1Loss(nn.Module):
    """
    Loss equivalent to nn.L1Loss with the following modifications: 

    1) mean along the temporal dimension is replaced by a discounted mean.
        The discounted mean for [x_0, x_1, ..., x_n] given discount factor p is 
        (x_0 + p*x_1 + ... + p^n*x_n)/(1 + p + ... + p^n). 

    2) option to mask loss
    """
    def __init__(self, discount=1.):
        """
        Args:
            discount (float): discount factor
        """
        super(DiscountedL1Loss, self).__init__()

        assert isinstance(discount, float), "discount factor must be a float"

        self.discount = discount

    @classmethod
    def factory(cls, config):
        """
        Create a DiscountedL1Loss instance from config.

        Args:
            config (addict): config object

        Returns: DiscountedL1Loss instance.
        """
        return cls(discount=config.train.discount)

    def forward(self, src, tgt, mask=None):
        """
        Compute discounted L1 loss.

        Args: 
            src (tensor): source data of shape [B, T, ...]

            tgt (tensor): target data of shape [B, T, ...]

            mask (tensor): (optional) binary mask of shape [B, T]

        Returns: scalar loss.
        """
        B, T = src.shape[0], src.shape[1]
        loss = torch.abs(tgt-src).view(B, T, -1)
        loss = torch.mean(loss, -1)
        coef = torch.pow(torch.full([T], self.discount).to(src.device), torch.arange(T).to(src.device))
        if mask is not None:
            coef = mask * coef
        return torch.mean(torch.sum(coef*loss, dim=-1)/torch.sum(coef, dim=-1))