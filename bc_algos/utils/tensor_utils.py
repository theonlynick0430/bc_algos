"""
A collection of utilities for working with nested tensor structures consisting
of numpy arrays and torch tensors.
"""
import collections
import numpy as np
import torch


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of 
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be 
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert(list not in type_func_dict)
    assert(tuple not in type_func_dict)
    assert(dict not in type_func_dict)

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError(
                'Cannot handle data type %s' % str(type(x)))
        
def pad_single(seq, dim, padding, pad_same=True, pad_values=None):
    """
    Pad input tensor or array @seq in the @dim dimension.

    Args:
        seq (np.ndarray or torch.Tensor): sequence to be padded
        dim (int): dimension in which to pad
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

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

    begin_pad = []
    end_pad = []

    if padding[0] > 0:
        pad = seq[[0]] if pad_same else ones_like_func(seq[[0]]) * pad_values
        begin_pad.append(repeat_func(pad, padding[0], dim))
    if padding[1] > 0:
        pad = seq[[-1]] if pad_same else ones_like_func(seq[[-1]]) * pad_values
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
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

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