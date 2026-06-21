from typing import Literal, Optional, SupportsIndex

import numba as nb
import numpy as np
from numba import types as nbt
from numba.extending import overload
from numpy import typing as npt

from .utils import (
    OMIT_OR_NONE,
    check_clipmode,
    check_out_shape,
    clip_if_outside,
    may_share_memory,
    raise_if_outside,
    replace_item_with_tuple,
    rollaxis,
    wrap_if_outside,
)


def _take_kernel(a, indices, axis, out, mode):
    check_clipmode(mode)

    a = rollaxis(a, axis)
    out = rollaxis(out, axis)

    out_buf = out
    if may_share_memory(a, out) or may_share_memory(indices, out):
        out_buf = np.empty_like(out)

    bound = len(a)
    if mode == "raise":
        for idx in nb.pndindex(indices.shape):
            i = raise_if_outside(indices[idx], bound)
            out_buf[idx] = a[i]

    if mode == "wrap":
        for idx in nb.pndindex(indices.shape):
            i = wrap_if_outside(indices[idx], bound)
            out_buf[idx] = a[i]

    if mode == "clip":
        for idx in nb.pndindex(indices.shape):
            i = clip_if_outside(indices[idx], bound)
            out_buf[idx] = a[i]

    if out is not out_buf:
        out[...] = out_buf


_take_kernel_inline = nb.njit(_take_kernel, boundscheck=False, inline="always")
_take_kernel_cached_parallel = nb.njit(
    _take_kernel, boundscheck=False, cache=True, parallel=True
)


def _take_kernel_1d_noncontiguous(a, indices, out, mode):
    check_clipmode(mode)

    out_buf = out
    if may_share_memory(a, out) or may_share_memory(indices, out):
        out_buf = np.empty_like(out)

    bound = a.size
    if mode == "raise":
        for idx in nb.pndindex(indices.shape):
            i = raise_if_outside(indices[idx], bound)
            out_buf[idx] = a.flat[i]

    if mode == "wrap":
        for idx in nb.pndindex(indices.shape):
            i = wrap_if_outside(indices[idx], bound)
            out_buf[idx] = a.flat[i]

    if mode == "clip":
        for idx in nb.pndindex(indices.shape):
            i = clip_if_outside(a[idx], bound)
            out_buf[idx] = a.flat[i]

    if out is not out_buf:
        out[...] = out_buf


def _take_kernel_1d_contiguous(a, indices, out, mode):
    check_clipmode(mode)

    out_buf = out
    if may_share_memory(a, out) or may_share_memory(indices, out):
        out_buf = np.empty_like(out)

    bound = a.size
    b = a.ravel()
    if mode == "raise":
        for idx in nb.pndindex(indices.shape):
            i = raise_if_outside(indices[idx], bound)
            out_buf[idx] = b[i]

    if mode == "wrap":
        for idx in nb.pndindex(indices.shape):
            i = wrap_if_outside(indices[idx], bound)
            out_buf[idx] = b[i]

    if mode == "clip":
        for idx in nb.pndindex(indices.shape):
            i = clip_if_outside(a[idx], bound)
            out_buf[idx] = b[i]

    if out is not out_buf:
        out[...] = out_buf


_take_kernel_1d_contiguous_inline = nb.njit(
    _take_kernel_1d_contiguous, boundscheck=False, inline="always"
)
_take_kernel_1d_contiguous_cached_parallel = nb.njit(
    _take_kernel_1d_contiguous, boundscheck=False, cache=True, parallel=True
)
_take_kernel_1d_noncontiguous_inline = nb.njit(
    _take_kernel_1d_noncontiguous, boundscheck=False, inline="always"
)
_take_kernel_1d_noncontiguous_cached_parallel = nb.njit(
    _take_kernel_1d_noncontiguous, boundscheck=False, cache=True, parallel=True
)


def take(
    a: npt.ArrayLike,
    indices: npt.ArrayLike,
    axis: Optional[int] = None,
    out: Optional[npt.NDArray] = None,
    mode: Optional[Literal["raise", "wrap", "clip"]] = "raise",
) -> npt.NDArray:
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
    a: npt.NDArray = np.asanyarray(a)
    indices: npt.NDArray = np.asanyarray(indices)

    if out is not None:
        out: npt.NDArray = np.asanyarray(out)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{indices.dtype=} must be subdtype of np.integer")

    if (axis is None) and (out is None) and a.flags.c_contiguous:
        out = np.empty(indices.shape, dtype=a.dtype)
        _take_kernel_1d_contiguous_cached_parallel(a, indices, out, mode)
        return out

    if (axis is None) and (out is None) and (not a.flags.c_contiguous):
        out = np.empty(indices.shape, dtype=a.dtype)
        _take_kernel_1d_noncontiguous_cached_parallel(a, indices, out, mode)
        return out

    if (axis is None) and isinstance(out, np.ndarray) and a.flags.c_contiguous:
        check_out_shape(out, indices.shape)
        _take_kernel_1d_contiguous_cached_parallel(a, indices, out, mode)
        return out

    if (axis is None) and isinstance(out, np.ndarray) and (not a.flags.c_contiguous):
        check_out_shape(out, indices.shape)
        _take_kernel_1d_noncontiguous_cached_parallel(a, indices, out, mode)
        return out

    if isinstance(axis, SupportsIndex) and (out is None):
        result_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
        out = np.empty(result_shape, dtype=a.dtype)
        _take_kernel_cached_parallel(a, indices, axis, out, mode)
        return out

    if isinstance(axis, SupportsIndex) and isinstance(out, np.ndarray):
        expected_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
        check_out_shape(out, expected_shape)
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

    if (
        isinstance(axis, OMIT_OR_NONE)
        and isinstance(out, OMIT_OR_NONE)
        and (a.layout == "C")
    ):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            out = np.empty(indices.shape, dtype=a.dtype)
            _take_kernel_1d_contiguous_inline(a, indices, out, mode)
            return out

        return impl

    if (
        isinstance(axis, OMIT_OR_NONE)
        and isinstance(out, OMIT_OR_NONE)
        and (a.layout != "C")
    ):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            out = np.empty(indices.shape, dtype=a.dtype)
            _take_kernel_1d_noncontiguous_inline(a, indices, out, mode)
            return out

        return impl

    if (
        isinstance(axis, OMIT_OR_NONE)
        and isinstance(out, nbt.Array)
        and (a.layout == "C")
    ):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            check_out_shape(out, indices.shape)
            _take_kernel_1d_contiguous_inline(a, indices, out, mode)
            return out

        return impl

    if (
        isinstance(axis, OMIT_OR_NONE)
        and isinstance(out, nbt.Array)
        and (a.layout != "C")
    ):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            check_out_shape(out, indices.shape)
            _take_kernel_1d_noncontiguous_inline(a, indices, out, mode)
            return out

        return impl

    if isinstance(axis, nbt.Integer) and isinstance(out, OMIT_OR_NONE):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            result_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
            out = np.empty(result_shape, dtype=a.dtype)
            _take_kernel_inline(a, indices, axis, out, mode)
            return out

        return impl

    if isinstance(axis, nbt.Integer) and isinstance(out, nbt.Array):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            expected_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
            check_out_shape(out, expected_shape)
            _take_kernel_inline(a, indices, axis, out, mode)
            return out

        return impl
    raise TypeError(
        f"{type(out)=} must be None or Array. {type(axis)=} must be None or int"
    )
