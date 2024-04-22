from bc_algos.models.base_nets import pos_enc_1d
from bc_algos.models.obs_nets import ObservationGroupEncoder, ActionDecoder
from bc_algos.models.backbone import Backbone, MLP, Transformer
from bc_algos.utils.constants import GoalMode
import bc_algos.utils.tensor_utils as TensorUtils
from torch import LongTensor
from abc import ABC, abstractmethod
import torch.nn as nn
import torch


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

    @abstractmethod
    def prepare(self, batch, device=None):
        """
        Prepare model input, target output, and loss mask by processing @batch.

        Args:
            batch (dict): nested dictionary returned from dataloader

            device: (optional) device to send tensors to

        Returns: model input (dict), target output (tensor), 
            and (optional) pad_mask (tensor).
        """
        return NotImplementedError


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

    def prepare(self, batch, device=None):
        """
        Prepare model input, target output, and loss mask by processing @batch.

        Args:
            batch (dict): nested dictionary returned from dataloader

            device: (optional) device to send tensors to

        Returns: model input (dict), target output (tensor), 
            and (optional) pad_mask (tensor).
        """
        batch = TensorUtils.to_tensor(x=batch, device=device)
        batch = TensorUtils.to_float(x=batch)
        action = batch["actions"]
        pad_mask = batch["pad_mask"] if "pad_mask" in batch else None
        return batch, action, pad_mask

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
    def __init__(self, obs_group_enc, backbone, action_dec, history, action_chunk, num_goal=None):
        """
        Args:
            obs_group_enc (ObservationGroupEncoder): input encoder

            backbone (Backbone): policy backbone

            action_dec (ActionDecoder): output decoder

            history (int): number of stacked frames provided as input to policy

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
        
        history = config.dataset.frame_stack
        action_chunk = config.dataset.seq_length
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
    
    def prepare(self, batch, device=None):
        """
        Prepare model input, target output, and loss mask by processing @batch.

        Args:
            batch (dict): nested dictionary returned from dataloader

            device: (optional) device to send tensors to

        Returns: model input (dict), target output (tensor), 
            and (optional) pad_mask (tensor).
        """
        batch = TensorUtils.to_tensor(x=batch, device=device)
        batch = TensorUtils.to_float(x=batch)
        batch["obs"] = TensorUtils.slice(x=batch["obs"], dim=1, start=0, end=self.history+1)
        action = batch["actions"]
        pad_mask = batch["pad_mask"] if "pad_mask" in batch else None
        return batch, action, pad_mask
    
    def create_pos_enc(self):
        """
        Create positional encodings for model input.
        """
        # self.embeddings = nn.Embedding(len(self.obs_group_enc.output_dim), self.embed_dim)

        # self.obs_group_to_pos_enc = nn.ParameterDict()
        # for obs_group in self.obs_group_enc.output_dim:
        #     T = self.num_goal if obs_group == "goal" else self.history+1
        #     N = self.obs_group_enc.output_dim[obs_group] // self.embed_dim
        #     pos_enc = pos_enc_1d(d_model=self.embed_dim, T=T)
        #     pos_enc = pos_enc.unsqueeze(1).repeat(1, N, 1).view(-1, self.embed_dim)
        #     pos_enc = nn.Parameter(pos_enc)
        #     pos_enc.requires_grad = False # buffer
        #     self.obs_group_to_pos_enc[obs_group] = pos_enc

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
        B = TensorUtils.get_batch_dim(input)
        latent_dict = TensorUtils.time_distributed(input=input, op=self.obs_group_enc)
        # src = []
        # for i, (obs_group, latent) in enumerate(latent_dict.items()):
        #     device = latent.device
        #     latent = latent.view(B, -1, self.embed_dim)
        #     embedding = self.embeddings(LongTensor([i]).to(device))
        #     pos_enc = self.obs_group_to_pos_enc[obs_group]
        #     src.append(latent + embedding + pos_enc)
        # src = torch.cat(src, dim=-2)
        src = torch.cat([latent.view(B, -1, self.embed_dim) for latent in latent_dict.values()], dim=-2)
        tgt = self.tgt.unsqueeze(0).repeat(B, 1, 1)
        output = self.backbone(src, tgt)
        action = TensorUtils.time_distributed(input=output, op=self.action_dec)
        return action
