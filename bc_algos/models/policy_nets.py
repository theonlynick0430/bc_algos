import torch.nn as nn

class BC(nn.Module):
    """
    A policy network that predicts actions from observations and goals using MLP.
    """
    