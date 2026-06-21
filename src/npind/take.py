from types import NoneType
from typing import Literal, Optional, SupportsIndex

import numba as nb
import numpy as np
from numba import types as nbt
from numba.extending import overload, register_jitable
from numpy import typing as npt

from .utils import (
    OMIT_OR_NONE_TYPES,
    check_indices_bounds,
    may_share_memory,
    modify_axis,
    replace_item_with_tuple,
    rollaxis,
    type_check,
    unravel_index,
)


@register_jitable(nopython=True)
def _modify_indices(indices, length, mode):
    if mode is None:
        return indices

    if mode == "raise":
        check_indices_bounds(indices, length)

    if mode == "warp":
        indices = indices % length

    if mode == "clip":
        np.clip(indices, 0, length - 1, out=indices)
    return indices


def _take_kernel(a, indices, axis, out, mode):
    axis = modify_axis(axis, a.ndim)
    indices = _modify_indices(indices, a.shape[axis], mode)

    a = rollaxis(a, axis)
    out = rollaxis(out, axis)

    out_buf = out
    if may_share_memory(a, out) or may_share_memory(indices, out):
        out_buf = np.empty_like(out)

    for idx in nb.pndindex(indices.shape):
        out_buf[idx] = a[indices[idx]]

    if out is not out_buf:
        out[...] = out_buf


_take_kernel_inline = nb.njit(_take_kernel, boundscheck=False, inline="always")
_take_kernel_cached_parallel = nb.njit(
    _take_kernel, boundscheck=False, cache=True, parallel=True
)


def _take_kernel_1d(a, indices, out, mode):
    indices = _modify_indices(indices, a.size, mode)

    out_buf = out
    if may_share_memory(a, out) or may_share_memory(indices, out):
        out_buf = np.empty_like(out)

    if a.flags.c_contiguous:
        b = a.ravel()
        for idx in nb.pndindex(indices.shape):
            out_buf[idx] = b[indices[idx]]
    else:
        for idx in nb.pndindex(indices.shape):
            a_idx = unravel_index(indices[idx], a.shape)
            out_buf[idx] = a[a_idx]

    if out is not out_buf:
        out[...] = out_buf


_take_kernel_1d_inline = nb.njit(_take_kernel_1d, boundscheck=False, inline="always")
_take_kernel_1d_cached_parallel = nb.njit(
    _take_kernel_1d, boundscheck=False, cache=True, parallel=True
)


def take(
    a: npt.ArrayLike,
    indices: npt.ArrayLike,
    axis: Optional[int] = None,
    out: Optional[np.ndarray] = None,
    mode: Optional[Literal["raise", "wrap", "clip"]] = "raise",
) -> np.ndarray:
    """
    Take elements from an array along an axis using parallel processing.

    This function provides a high-performance alternative to `numpy.take`,
    leveraging Numba's multi-core parallelization. It is particularly efficient
    when an output buffer is provided and the memory is contiguous.

    Parameters
    ----------
    a : array_like
        The source array.
    indices : array_like
        The indices of the values to extract.
    axis : int, optional
        The axis over which to select values. By default, None
    out : ndarray, optional
        If provided, the result will be placed into this array. It should
        be of the appropriate shape and dtype.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.
        Currently only 'raise' is supported, which raises an IndexError.

    Returns
    -------
    out : ndarray
        The returned array has the same type as `a`.

    Raises
    ------
    IndexError
        If indices are out of bounds.
    ValueError
        If the shape of `out` does not match the expected result shape.

    Notes
    -----
    The acceleration is powered by Numba's `@njit(parallel=True)`.
    For optimal performance, ensure that `a`, `indices`, and `out`
    are C-contiguous.

    Examples
    --------
    >>> import npind as npi
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> npi.take(a, [0, 1], axis=1)
    array([[1, 2],
           [3, 4]])
    """
    a: np.ndarray = np.asanyarray(a)
    indices: np.ndarray = np.asanyarray(indices)

    if out is not None:
        out: np.ndarray = np.asanyarray(out)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{indices.dtype=} must be subdtype of np.integer")

    if type_check((axis, out), (NoneType, NoneType)):
        out = np.empty(indices.shape, dtype=a.dtype)
        _take_kernel_1d_cached_parallel(a, indices, out, mode)
        return out
    if type_check((axis, out), (NoneType, np.ndarray)):
        if out.shape != indices.shape:
            raise TypeError("shape of out-array does not match result of take")
        _take_kernel_1d_cached_parallel(a, indices, out, mode)
        return out
    if type_check((axis, out), (SupportsIndex, NoneType)):
        result_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
        out = np.empty(result_shape, dtype=a.dtype)
        _take_kernel_cached_parallel(a, indices, axis, out, mode)
        return out
    if type_check((axis, out), (SupportsIndex, np.ndarray)):
        expected_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
        if out.shape != expected_shape:
            raise TypeError("shape of out-array does not match result of take")
        _take_kernel_cached_parallel(a, indices, axis, out, mode)
        return out

    raise TypeError(
        f"{type(out).__name__=} must be None or Array. {type(axis).__name__=} must be None or int"
    )


@overload(take, inline="always")
def overload_take(a, indices, axis=None, out=None, mode="raise"):
    if isinstance(axis, nbt.Optional):
        axis = axis.type
    if isinstance(out, nbt.Optional):
        out = out.type

    if not isinstance(a, nbt.Array):
        raise TypeError(f"{type(a).__name__=} must be Array")
    if not isinstance(indices, nbt.Array):
        raise TypeError(f"{type(indices).__name__=} must be Array")

    if type_check((axis, out), (OMIT_OR_NONE_TYPES, OMIT_OR_NONE_TYPES)):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            out = np.empty(indices.shape, dtype=a.dtype)
            _take_kernel_1d_inline(a, indices, out, mode)
            return out

        return impl
    if type_check((axis, out), (OMIT_OR_NONE_TYPES, nbt.Array)):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            expected_shape = indices.shape
            if out.shape != expected_shape:
                raise TypeError("shape of out-array does not match result of take")

            _take_kernel_1d_inline(a, indices, out, mode)
            return out

        return impl
    if type_check((axis, out), (nbt.Integer, OMIT_OR_NONE_TYPES)):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            result_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
            out = np.empty(result_shape, dtype=a.dtype)
            _take_kernel_inline(a, indices, axis, out, mode)
            return out

        return impl
    if type_check((axis, out), (nbt.Integer, nbt.Array)):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            expected_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
            if out.shape != expected_shape:
                raise TypeError("shape of out-array does not match result of take")

            _take_kernel_inline(a, indices, axis, out, mode)
            return out

        return impl
    raise TypeError(
        f"{type(out)=} must be None or Array. {type(axis)=} must be None or int"
    )
