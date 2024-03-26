import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


def run_epoch(model, data_loader, loss_fn, optimizer=None, validate=False):
    """
    Run a single epoch of training/validation by iterating 
    fully through data loader. If @validate is False, update model weights
    according to @loss_fn and @optimizer.

    Args: 
        model (nn.Module): network to train/validate

        data_loader (DataLoader): data source

        loss_fn (nn.Module): function to compute loss

        optimizer (optim.Optimizer): (optional) function to update model parameters.

        validate (bool): If False, train model on data. If True, validate model on data. 
            Defaults to False.

    Returns: avg loss
    """
    assert isinstance(model, nn.Module)
    assert isinstance(data_loader, DataLoader)
    assert isinstance(optimizer, optim.Optimizer)
    assert isinstance(loss_fn, nn.Module)

    if not validate:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.
    with tqdm(total=len(data_loader), unit='batch') as progress_bar:
        for batch in data_loader:
            
            outputs = model(batch)
            
            # Compute loss
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.update(1)
        
        # Print average loss for the epoch
        print(f"Average Loss: {total_loss / len(data_loader)}")