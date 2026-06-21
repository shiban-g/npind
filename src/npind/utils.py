from types import NoneType

import numpy as np
from numba import types as nbt
from numba.extending import register_jitable

OMIT_OR_NONE_TYPES = (nbt.Omitted, nbt.NoneType, NoneType)


def type_check(variablea, types):
    return all(isinstance(v, t) for v, t in zip(variablea, types))


@register_jitable(nopython=True)
def rollaxis(a: np.ndarray, n: int):
    axes = a.shape
    for i in range(a.ndim):
        axes = axes[1:] + (i,)
    for i in range(n):
        axes = axes[1:] + axes[:1]
    return a.transpose(axes)


@register_jitable(nopython=True)
def may_share_memory(a, b):
    a_ptr = a.ctypes.data
    b_ptr = b.ctypes.data

    st, sh = (
        max(zip(a.strides, a.shape)) if a_ptr <= b_ptr else max(zip(b.strides, b.shape))
    )
    return abs(a_ptr - b_ptr) < st * sh


@register_jitable(nopython=True)
def check_indices_bounds(indices, length):
    if indices.size == 0:
        return
    min_ = indices.min()
    max_ = indices.max()
    if not (0 <= min_ <= max_ < length):
        raise IndexError("indices are out of bounds")


@register_jitable(nopython=True)
def modify_axis(axis, ndim):
    if not (-ndim <= axis < ndim):
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    if axis < 0:
        axis += ndim
    return axis


@register_jitable(nopython=True)
def rotate_left(tup, n):
    """Rotates a tuple to the left by n positions."""
    for _ in range(n):
        tup = tup[1:] + tup[:1]
    return tup


@register_jitable(nopython=True)
def rotate_right(tup, n):
    """Rotates a tuple to the right by n positions."""
    for _ in range(n):
        tup = tup[-1:] + tup[:-1]
    return tup


@register_jitable(nopython=True)
def replace_item_with_tuple(target_tuple, replacement_tuple, index):
    """
    Replaces the element at index in `target_tuple` with `replacement_tuple`.

    >>> replace_item_with_tuple((1, 2, 3), (4, 5, 6), 1)
    (1, 4, 5, 6, 3)
    """
    rotated_tuple = rotate_left(target_tuple, index)
    temp_result = replacement_tuple + rotated_tuple[1:]
    result = rotate_right(temp_result, index)
    return result


@register_jitable(nopython=True)
def replace_items(target_tuple, replacement, index):
    """
    Replaces the element at index in `target_tuple` with `replacement`.

    >>> replace_item_with_tuple((1, 2, 3), 4, 1)
    (1, 4, 3)
    """
    return replace_item_with_tuple(target_tuple, (replacement,), index)


@register_jitable(nopython=True)
def unravel_index(flat_index, shape):
    """
    A Numba-jitable equivalent of numpy.unravel_index.
    Converts a flat index into a tuple of coordinate arrays.
    Only 'C' order is supported.
    """
    for _ in range(len(shape)):
        flat_index, i = divmod(flat_index, shape[-1])
        shape = (i,) + shape[:-1]
    return shape
