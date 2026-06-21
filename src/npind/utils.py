from types import NoneType

import numba as nb
import numpy.typing as npt
from numba import types as nbt
from numba.extending import register_jitable

OMIT_OR_NONE = (nbt.Omitted, nbt.NoneType, NoneType)


@register_jitable(nopython=True)
def check_clipmode(mode):
    if mode not in ("raise", "wrap", "clip"):
        raise ValueError("mode must be one of 'clip', 'raise', or 'wrap'")


@register_jitable(nopython=True)
def check_out_shape(array: npt.NDArray, shape: tuple):
    if array.shape != shape:
        raise ValueError("shape of out-array does not match result")


@nb.njit(inline="always")
def raise_if_outside(index: int, bound: int) -> int:
    if 0 <= index < bound:
        return index
    raise IndexError("indices are out of bounds")


@nb.njit(inline="always")
def wrap_if_outside(index: int, bound: int) -> int:
    if 0 <= index < bound:
        return index
    return index % bound


@nb.njit(inline="always")
def clip_if_outside(index: int, bound: int) -> int:
    if index < 0:
        return 0
    if bound <= index:
        return bound - 1
    return index


@register_jitable(nopython=True)
def may_share_memory(a: npt.NDArray, b: npt.NDArray) -> bool:
    a_ptr = a.ctypes.data
    b_ptr = b.ctypes.data

    st, sh = (
        max(zip(a.strides, a.shape)) if a_ptr <= b_ptr else max(zip(b.strides, b.shape))
    )
    return abs(a_ptr - b_ptr) < st * sh


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
def rollaxis(a: npt.NDArray, n: int) -> npt.NDArray:
    if n == 0:
        return a

    axes = a.shape
    for i in range(a.ndim):
        axes = axes[1:] + (i,)
    if n > 0:
        axes = rotate_left(axes, n)
    if n < 0:
        axes = rotate_right(axes, -n)
    return a.transpose(axes)
