from bc_algos.models.base_nets import pos_enc_1d
from bc_algos.models.obs_nets import ObservationGroupEncoder, ActionDecoder
from bc_algos.models.backbone import Backbone, MLP, Transformer
from bc_algos.utils.constants import GoalMode
import bc_algos.utils.tensor_utils as TensorUtils
from torch import LongTensor
import torch.nn as nn
import torch
from abc import ABC


class BC(ABC, nn.Module):
    """
    Abstract class for behavorial cloning policy that predicts actions from observations.
    Subclass to implement different behavorial cloning policies.
    """
    def __init__(self, obs_group_enc, backbone, action_dec):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            action_dec (ActionDecoder): output decoder
        """
        super(BC, self).__init__()

        assert isinstance(obs_group_enc, ObservationGroupEncoder)
        assert isinstance(backbone, Backbone)
        assert isinstance(action_dec, ActionDecoder)

        self.obs_group_enc = obs_group_enc
        self.backbone = backbone
        self.action_dec = action_dec

    @classmethod
    def prepare_input(cls, input, device=None):
        """
        Prepare input to be processed by model by converting to float tensor
        and moving to specified device.

        Args:
            input (dict): nested dictionary that maps observation group to observation key
                to data (tensor) of shape [B, ...]

            device: (optional) device to send tensors to

        Returns: prepared input.
        """
        input = TensorUtils.to_tensor(x=input, device=device)
        return TensorUtils.to_float(x=input)


class BC_MLP(BC):
    """
    Module for behavorial cloning policy that predicts actions from observations using MLP.
    """
    def __init__(self, obs_group_enc, backbone, action_dec):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            action_dec (ActionDecoder): output decoder
        """
        super(BC_MLP, self).__init__(obs_group_enc=obs_group_enc, backbone=backbone, action_dec=action_dec)

        assert isinstance(backbone, MLP)

    def forward(self, input):
        """
        Forward pass through BC_MLP.

        Args: 
            input (dict): nested dictionary that maps observation group to observation key
                to data (tensor) of shape [B, T=1, ...]

        Returns: action (tensor) of shape [B, T=1, action_dim].
        """
        latent_dict = TensorUtils.time_distributed(input=input, op=self.obs_group_enc)
        latents = torch.cat(list(latent_dict.values()), dim=-1)
        action = TensorUtils.time_distributed(input=latents, op=nn.Sequential(self.backbone, self.action_dec))
        return action
    

class BC_Transformer(BC):
    """
    Module for behavorial cloning policy that predicts actions from observations 
    using transformer encoder-decoder architecture.
    """
    def __init__(
        self, 
        obs_group_enc, 
        backbone, 
        action_dec, 
        history, 
        action_chunk, 
        num_goal=None,
    ):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            action_dec (ActionDecoder): output decoder

            history (int): number of frames provided as input to policy as history

            action_chunk (int): number of actions to predict in a single model pass

            num_goal (GoalMode): (optional) number of goals provided as input to policy
        """
        super(BC_Transformer, self).__init__(obs_group_enc=obs_group_enc, backbone=backbone, action_dec=action_dec)

        assert isinstance(backbone, Transformer)

        self.history = history
        self.action_chunk = action_chunk
        self.num_goal = num_goal
        self.embed_dim = backbone.embed_dim

        self.create_pos_enc()

    @classmethod
    def factory(cls, config, obs_group_enc, backbone, action_dec):
        """
        Create a BC_Transformer instance from config.

        Args:
            config (addict): config object

            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            action_dec (ActionDecoder): output decoder

        Returns: BC_Transformer instance.
        """
        
        history = config.dataset.history
        action_chunk = config.dataset.action_chunk
        goal_mode = config.dataset.goal_mode
        num_subgoal = config.dataset.num_subgoal

        num_goal = None
        if goal_mode == GoalMode.LAST:
            num_goal = 1
        elif goal_mode == GoalMode.SUBGOAL:
            num_goal = action_chunk
        elif goal_mode == GoalMode.FULL:
            num_goal = num_subgoal

        return cls(
            obs_group_enc=obs_group_enc,
            backbone=backbone,
            action_dec=action_dec,
            history=history,
            action_chunk=action_chunk,
            num_goal=num_goal,
        )
    
    def create_pos_enc(self):
        """
        Create positional encodings for model input.
        """
        self.embedding = nn.Embedding(1, self.embed_dim)

        T = self.num_goal
        N = self.obs_group_enc.output_dim["goal"] // self.embed_dim
        goal_pos_enc = pos_enc_1d(d_model=self.embed_dim, T=T)
        goal_pos_enc = goal_pos_enc.unsqueeze(1).repeat(1, N, 1).view(-1, self.embed_dim)
        self.goal_pos_enc = nn.Parameter(goal_pos_enc)
        self.goal_pos_enc.requires_grad = False # buffer

        self.tgt = nn.Parameter(pos_enc_1d(d_model=self.embed_dim, T=self.action_chunk))
        self.tgt.requires_grad = False # buffer

    def forward(self, input):
        """
        Forward pass through BC_Transformer.

        Args: 
            input (dict): nested dictionary that maps observation group to observation key
                to data (tensor) of shape [B, T_obs/T_goal, ...]

        Returns: action (tensor) of shape [B, T=@self.action_chunk, action_dim].
        """
        B = TensorUtils.get_batch_dim(x=input)
        latent_dict = TensorUtils.time_distributed(input=input, op=self.obs_group_enc)
        obs_latent = latent_dict["obs"].view(B, -1, self.embed_dim)
        goal_latent = latent_dict["goal"].view(B, -1, self.embed_dim)
        goal_embedding = self.embedding(LongTensor([0]).to(goal_latent.device))
        goal_latent = goal_latent + goal_embedding + self.goal_pos_enc
        src = torch.cat([obs_latent, goal_latent], dim=-2)
        tgt = self.tgt.unsqueeze(0).repeat(B, 1, 1)
        output = self.backbone(src, tgt)
        action = TensorUtils.time_distributed(input=output, op=self.action_dec)
        return action
