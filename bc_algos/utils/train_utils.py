import bc_algos.utils.tensor_utils as TensorUtils
from bc_algos.models.policy_nets import BC
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import wandb


def run_epoch(model, data_loader, loss_fn, history, optimizer=None, validate=False, device=None):
    """
    Run a single epoch of training/validation by iterating 
    fully through data loader. If @validate is False, update model weights
    according to @loss_fn and @optimizer.

    Args: 
        model (nn.Module): network to train/validate

        data_loader (DataLoader): data source

        loss_fn (nn.Module): function to compute loss

        history (int): number of frames provided as input to policy as history

        optimizer (optim.Optimizer): (optional) function to update model parameters.

        validate (bool): If False, train model on data. If True, validate model on data. 
            Defaults to False.

        device: (optional) device to send tensors to
    """
    assert isinstance(model, nn.Module)
    assert isinstance(data_loader, DataLoader)
    assert isinstance(optimizer, optim.Optimizer)
    assert isinstance(loss_fn, nn.Module)

    if not validate:
        model.train()
    else:
        model.eval()
    
    with tqdm(total=len(data_loader), unit='batch') as progress:
        for batch in data_loader:
            # prepare input and target
            batch = BC.prepare_input(input=batch, device=device)
            target = batch[data_loader.dataset.action_key][:, history:, :]
            pad_mask = None
            if data_loader.dataset.get_pad_mask is True:
                pad_mask = batch["pad_mask"][:, history:]
            batch["obs"] = TensorUtils.slice(x=batch["obs"], dim=1, start=0, end=history+1)
            # generate output
            output = model(batch)
            # compute loss
            loss = loss_fn(output, target, pad_mask)
            if not validate:
                wandb.log({"train_loss": loss.item()})
                # if training, update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                wandb.log({"valid_loss": loss.item()})
            progress.update(1)