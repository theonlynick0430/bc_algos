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

def map_tensor(x, func):
    """
    Apply function @func to torch.Tensor objects in a nested dictionary or
    list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        func (function): function to apply to each tensor

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: func,
            type(None): lambda x: x,
        }
    )

def join_dimensions(x, begin_axis, end_axis):
    """
    Joins all dimensions between dimensions (@begin_axis, @end_axis) into a flat dimension, for
    all tensors in nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]),
            np.ndarray: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]),
            type(None): lambda x: x,
        }
    )

def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions in a tensor to a target dimension.

    Args:
        x (torch.Tensor): tensor to reshape
        begin_axis (int): begin dimension
        end_axis (int): end dimension (inclusive)
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (torch.Tensor): reshaped tensor
    """
    assert(begin_axis <= end_axis)
    assert(begin_axis >= 0)
    assert(end_axis < len(x.shape))
    assert(isinstance(target_dims, (tuple, list)))
    s = x.shape
    final_s = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)

def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions for all tensors in nested dictionary or list or tuple 
    to a target dimension.
    
    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension (inclusive)
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=t),
            np.ndarray: lambda x, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=t),
            type(None): lambda x: x,
        }
    )

def flatten_nested_dict_list(d, parent_key='', sep='_', item_key=''):
    """
    Flatten a nested dict or list to a list.

    For example, given a dict
    {
        a: 1
        b: {
            c: 2
        }
        c: 3
    }

    the function would return [(a, 1), (b_c, 2), (c, 3)]

    Args:
        d (dict, list): a nested dict or list to be flattened
        parent_key (str): recursion helper
        sep (str): separator for nesting keys
        item_key (str): recursion helper
    Returns:
        list: a list of (key, value) tuples
    """
    items = []
    if isinstance(d, (tuple, list)):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for i, v in enumerate(d):
            items.extend(flatten_nested_dict_list(v, new_key, sep=sep, item_key=str(i)))
        return items
    elif isinstance(d, dict):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for k, v in d.items():
            assert isinstance(k, str)
            items.extend(flatten_nested_dict_list(v, new_key, sep=sep, item_key=k))
        return items
    else:
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        return [(new_key, d)]

def time_distributed(inputs, op, activation=None, inputs_as_kwargs=False, inputs_as_args=False, **kwargs):
    """
    Apply function @op to all tensors in nested dictionary or list or tuple @inputs in both the
    batch (B) and time (T) dimension, where the tensors are expected to have shape [B, T, ...].
    Will do this by reshaping tensors to [B * T, ...], passing through the op, and then reshaping
    outputs to [B, T, ...].

    Args:
        inputs (list or tuple or dict): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        op: a layer op that accepts inputs
        activation: activation to apply at the output
        inputs_as_kwargs (bool): whether to feed input as a kwargs dict to the op
        inputs_as_args (bool) whether to feed input as a args list to the op
        kwargs (dict): other kwargs to supply to the op

    Returns:
        outputs (dict or list or tuple): new nested dict-list-tuple with tensors of leading dimension [B, T].
    """
    batch_size, seq_len = flatten_nested_dict_list(inputs)[0][1].shape[:2]
    inputs = join_dimensions(inputs, 0, 1)
    if inputs_as_kwargs:
        outputs = op(**inputs, **kwargs)
    elif inputs_as_args:
        outputs = op(*inputs, **kwargs)
    else:
        outputs = op(inputs, **kwargs)

    if activation is not None:
        outputs = map_tensor(outputs, activation)
    outputs = reshape_dimensions(outputs, begin_axis=0, end_axis=0, target_dims=(batch_size, seq_len))
    return outputs

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

def to_tensor(x, device=None):
    """
    Converts all numpy arrays in nested dictionary or list or tuple to
    torch tensors (and leaves existing torch Tensors as-is), and returns 
    a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        
        device: device to send tensors to

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.to(device=device),
            np.ndarray: lambda x: torch.from_numpy(x).to(device=device),
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