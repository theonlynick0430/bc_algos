import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pos_enc_1d(d_model, T):
    """
    Returns 1D positional encodings for transformer model.

    Args: 
        d_model (int): embedding dim

        T (int): temporal dim
    """
    if d_model % 2 != 0:
        raise ValueError("cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    
    pe = torch.zeros(T, d_model)

    position = torch.arange(0, T).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def pos_enc_2d(d_model, H, W):
    """
    Returns 2D positional encodings for transformer model.

    Args: 
        d_model (int): embedding dim

        H (int): height

        W (int): width
    """
    if d_model % 4 != 0:
        raise ValueError("cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    
    pe = torch.zeros(d_model, H, W)

    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., W).unsqueeze(1)
    pos_h = torch.arange(0., H).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)

    return pe


class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in [1].
    Concretely, the spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.
    References:
        [1]: End-to-End Training of Deep Visuomotor Policies,
        https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize=True):
        """Constructor.
        Args:
            normalize (bool): if True, use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w).to(device),
                    torch.linspace(-1, 1, h).to(device),
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w).to(device),
                torch.arange(0, h).to(device),
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        xc, yc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1)
        y_mean = (softmax * yc.flatten()).sum(dim=1)

        # concatenate and reshape the result
        # to (B, 2, C) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.stack((x_mean.view(b, c), y_mean.view(b, c)), dim=1)