"""
A collection of utilities for working with nested tensor structures consisting
of numpy arrays and torch tensors.
"""
from collections import OrderedDict
import numpy as np
import torch


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of 
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

        type_func_dict (dict): a mapping from data types to the functions to be 
            applied for each data type

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert(list not in type_func_dict)
    assert(tuple not in type_func_dict)
    assert(dict not in type_func_dict)

    if isinstance(x, (dict, OrderedDict)):
        y = { k: recursive_dict_list_tuple_apply(v, type_func_dict) for k,v in x.items() }
        if isinstance(x, OrderedDict):
            return OrderedDict(y)
        else:
            return y
    elif isinstance(x, (list, tuple)):
        y = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            return tuple(y)
        else:
            return y
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError('Cannot handle data type %s' % str(type(x)))
        
def pad_single(seq, dim, padding, pad_same=True, pad_values=None):
    """
    Pad input tensor or array @seq in the @dim dimension.

    Args:
        seq (np.ndarray or torch.Tensor): sequence to be padded

        dim (int): dimension in which to pad

        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1

        pad_same (bool): if pad by duplicating

        pad_values (scalar or (ndarray, Tensor)): (optional) values to be padded if not pad_same

    Returns:
        padded sequence (np.ndarray or torch.Tensor)
    """
    assert isinstance(seq, (np.ndarray, torch.Tensor))
    assert pad_same or pad_values is not None
    if pad_values is not None:
        assert isinstance(pad_values, float)
    repeat_func = np.repeat if isinstance(seq, np.ndarray) else torch.repeat_interleave
    concat_func = np.concatenate if isinstance(seq, np.ndarray) else torch.cat
    ones_like_func = np.ones_like if isinstance(seq, np.ndarray) else torch.ones_like
    device_kwargs = {} if isinstance(seq, np.ndarray) else {"device": seq.device}

    begin_pad = []
    end_pad = []

    if padding[0] > 0:
        pad = seq[[0]] if pad_same else ones_like_func(seq[[0]], *device_kwargs) * pad_values
        begin_pad.append(repeat_func(pad, padding[0], dim))
    if padding[1] > 0:
        pad = seq[[-1]] if pad_same else ones_like_func(seq[[-1]], *device_kwargs) * pad_values
        end_pad.append(repeat_func(pad, padding[1], dim))

    return concat_func(begin_pad + [seq] + end_pad, dim)

def pad(seq, dim, padding, pad_same=True, pad_values=None):
    """
    Pad a nested dictionary or list or tuple of sequence tensors @seq in @dim dimension.

    Args:
        seq (dict or list or tuple): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]

        dim (int): dimension in which to pad

        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1

        pad_same (bool): if pad by duplicating
        
        pad_values (scalar or (ndarray, Tensor)): (optional) values to be padded if not pad_same

    Returns:
        padded sequence (dict or list or tuple)
    """
    return recursive_dict_list_tuple_apply(
        seq,
        {
            torch.Tensor: lambda x, d=dim, p=padding, ps=pad_same, pv=pad_values:
                pad_single(x, d, p, ps, pv),
            np.ndarray: lambda x, d=dim, p=padding, ps=pad_same, pv=pad_values:
                pad_single(x, d, p, ps, pv),
            type(None): lambda x: x,
        }
    )

def get_batch_dim(x):
    """
    Find batch dim of data in nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
            of data of shape [B, ...]

    Returns:
        B (int): batch dim
    """
    if isinstance(x, (tuple, list)):
        return get_batch_dim(x[0])
    elif isinstance(x, dict):
        return get_batch_dim(list(x.values())[0])
    else:
        return x.shape[0]

def time_distributed(input, op):
    """
    Apply function @op to all tensors in nested dictionary or list or tuple @input in both the
    batch (B) and time (T) dim. Tensors are expected to have shape [B, T, ...], where T can vary
    between different tensors. Accomplishes this by reshaping tensors to [B * T, ...], 
    passing through the op, and then reshaping output to [B, T, ...].

    Args:
        input (dict or list or tuple): a possibly nested dictionary or list or tuple
            of data of shape [B, T, ...], where T can vary between different tensors

        op (nn.Module): layer that accepts x as input

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    B = get_batch_dim(input)

    merged = recursive_dict_list_tuple_apply(
        input,
        {
            torch.Tensor: lambda x: x.view(-1, *x.shape[2:]),
            np.ndarray: lambda x: x.reshape(-1, *x.shape[2:]),
            type(None): lambda x: x,
        }
    )

    output = op(merged)

    return recursive_dict_list_tuple_apply(
        output,
        {
            torch.Tensor: lambda x: x.view(B, -1, *x.shape[1:]),
            np.ndarray: lambda x: x.reshape(B, -1, *x.shape[1:]),
            type(None): lambda x: x,
        }
    )

def to_batch(x):
    """
    Introduces a leading batch dimension of 1 for all torch tensors and numpy 
    arrays in nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x[None, ...],
            np.ndarray: lambda x: x[None, ...],
            type(None): lambda x: x,
        }
    )

def to_sequence(x):
    """
    Introduces a time dimension of 1 at dimension 1 for all torch tensors and numpy 
    arrays in nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x[:, None, ...],
            np.ndarray: lambda x: x[:, None, ...],
            type(None): lambda x: x,
        }
    )

def repeat_seq(x, k):
    """
    Repeats the input along dimension 1 for torch tensors and numpy arrays in
    nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

        k (int): number of times to repeat along dim 1

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: torch.repeat_interleave(x, k, dim=1),
            np.ndarray: lambda x: np.repeat(x, k, axis=1),
            type(None): lambda x: x,
        }
    )

def shift_seq(x, k):
    """
    Shifts the input by along dimension 1 for torch tensors and numpy arrays in 
    nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

        k (int): shift along dim 1

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: torch.roll(x, k, dims=1),
            np.ndarray: lambda x: np.roll(x, k, axis=1),
            type(None): lambda x: x,
        }
    )

def slice(x, dim, start, end):
    """
    Slice the input by along dimension @dim from indices @start to @end 
    for torch tensors and numpy arrays in nested dictionary or list or tuple 
    and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

        dim (int): dimension to slice along

        start (int): start index

        end (int): end index

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: torch.index_select(x, dim, torch.arange(start, end, device=x.device)),
            np.ndarray: lambda x: np.take(x, np.arange(start, end), axis=dim),
            type(None): lambda x: x,
        }
    )

def to_numpy(x):
    """
    Converts all torch tensors in nested dictionary or list or tuple to
    numpy arrays (and leaves existing numpy arrays as-is), and returns 
    a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.detach().cpu().numpy(),
            np.ndarray: lambda x: x,
            type(None): lambda x: x,
        }
    )

def to_tensor(x, device=None):
    """
    Converts all numpy arrays in nested dictionary or list or tuple to
    torch tensors (and leaves existing torch Tensors as-is), and returns 
    a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        
        device: (optional) device to send tensors to

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.to(device),
            np.ndarray: lambda x: torch.from_numpy(x).to(device),
            type(None): lambda x: x,
        }
    )

def to_float(x):
    """
    Converts all torch tensors and numpy arrays in nested dictionary or list 
    or tuple to float type entries, and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.float(),
            np.ndarray: lambda x: x.astype(np.float32),
            type(None): lambda x: x,
        }
    )