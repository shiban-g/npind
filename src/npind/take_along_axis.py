from typing import Optional, SupportsIndex

import numba as nb
import numpy as np
from numba import types as nbt
from numba.extending import overload, register_jitable
from numpy import typing as npt

from .utils import (
    OMIT_OR_NONE,
    may_share_memory,
    modify_axis,
    raise_if_outside,
)


@register_jitable(nopython=True)
def _check_shapes(a, indices, axis):
    if a.ndim != indices.ndim:
        raise ValueError("`a` and `indices` must have the same number of dimensions")
    for i in range(a.ndim):
        if i != axis and a.shape[i] != indices.shape[i]:
            raise ValueError("Dimension mismatch")


@register_jitable(nopython=True)
def _check_indices_1d(indices):
    if indices.ndim != 1:
        raise ValueError("when axis=None, `indices` must have a single dimension.")


@register_jitable(nopython=True)
def _check_out_shape(indices, out):
    if out.shape != indices.shape:
        raise ValueError("shape of out-array does not match shape of indices")


def _take_along_axis_kernel(a, indices, axis, out):
    axis = modify_axis(axis, a.ndim)

    a = np.swapaxes(a, 0, axis)
    indices = np.swapaxes(indices, 0, axis)
    out = np.swapaxes(out, 0, axis)

    out_buf = out
    if may_share_memory(a, out) or may_share_memory(indices, out):
        out_buf = np.empty_like(out)

    bound = a.shape[0]
    for idx in nb.pndindex(indices.shape):
        i = raise_if_outside(indices[idx], bound)
        out_buf[idx] = a[(i,) + idx[1:]]

    if out is not out_buf:
        out[...] = out_buf


def _take_along_axis_kernel_1d_noncontiguous(a, indices, out):
    out_buf = out
    if may_share_memory(a, out) or may_share_memory(indices, out):
        out_buf = np.empty_like(out)

    bound = a.size
    for idx in nb.prange(indices.size):
        i = raise_if_outside(indices[idx], bound)
        out_buf[idx] = a.flat[i]

    if out is not out_buf:
        out[...] = out_buf


def _take_along_axis_kernel_1d_contiguous(a, indices, out):
    out_buf = out
    if may_share_memory(a, out) or may_share_memory(indices, out):
        out_buf = np.empty_like(out)

    bound = a.size
    b = a.ravel()
    for idx in nb.prange(indices.size):
        i = raise_if_outside(indices[idx], bound)
        out_buf[idx] = b[i]

    if out is not out_buf:
        out[...] = out_buf


_take_along_axis_kernel_inline = nb.njit(
    _take_along_axis_kernel, boundscheck=False, inline="always"
)
_take_along_axis_kernel_cached_parallel = nb.njit(
    _take_along_axis_kernel, boundscheck=False, cache=True, parallel=True
)
_take_along_axis_kernel_1d_contiguous_inline = nb.njit(
    _take_along_axis_kernel_1d_contiguous, boundscheck=False, inline="always"
)
_take_along_axis_kernel_1d_contiguous_cached_parallel = nb.njit(
    _take_along_axis_kernel_1d_contiguous, boundscheck=False, cache=True, parallel=True
)
_take_along_axis_kernel_1d_noncontiguous_inline = nb.njit(
    _take_along_axis_kernel_1d_noncontiguous, boundscheck=False, inline="always"
)
_take_along_axis_kernel_1d_noncontiguous_cached_parallel = nb.njit(
    _take_along_axis_kernel_1d_noncontiguous,
    boundscheck=False,
    cache=True,
    parallel=True,
)


def take_along_axis(
    a: npt.ArrayLike,
    indices: npt.ArrayLike,
    axis: Optional[int] = None,
    out: Optional[npt.NDArray] = None,
) -> npt.NDArray:
    """
    Take values from the input array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis
    in the index and data arrays, and uses the former to look up values in
    the latter. These slices can be different lengths.

    Functions returning an index along an axis, like `argsort` and
    `argpartition`, produce suitable indices for this function.

    This function provides a high-performance alternative to
    `numpy.take_along_axis`, leveraging Numba's multi-core parallelization.

    Parameters
    ----------
    a : array_like
        Source array.
    indices : array_like
        Indices to take along each 1d slice of `a`. This must have the same
        number of dimensions as `a`. All dimensions except `axis` must match
        those of `a`.
    axis : int or None, optional
        The axis to take 1d slices along. If axis is None, the input array
        is treated as if it had first been flattened to 1d, for consistency
        with `sort` and `argsort`. In this case, `indices` must be
        one-dimensional.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must
        have a shape that matches `indices`.

    Returns
    -------
    out : ndarray
        The indexed result. The returned array has the same type as `a`.

    Raises
    ------
    IndexError
        If an index is out of bounds for the corresponding 1d slice.
    ValueError
        If `a` and `indices` do not have compatible shapes, or if the shape
        of `out` does not match that of `indices`.

    See Also
    --------
    take : Take along an axis, using the same indices for every 1d slice.
    numpy.take_along_axis : NumPy reference implementation.
    numpy.put_along_axis : Put values into the destination array by matching
        1d index and data slices.

    Notes
    -----
    For a given axis, this is equivalent to (but faster than) iterating over
    matching 1d slices of `a` and `indices` and assigning
    ``out_1d[:] = a_1d[indices_1d]``.

    The acceleration is powered by Numba's ``@njit(parallel=True)``.
    For optimal performance, ensure that `a`, `indices`, and `out`
    are C-contiguous.

    Examples
    --------
    >>> import npind as npi
    >>> import numpy as np
    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    We can sort either by using `sort` directly, or `argsort` and this
    function:

    >>> np.sort(a, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])
    >>> ai = np.argsort(a, axis=1)
    >>> npi.take_along_axis(a, ai, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])

    The same works for max and min, if you maintain the trivial dimension
    with `keepdims`:

    >>> np.max(a, axis=1, keepdims=True)
    array([[30],
           [60]])
    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> npi.take_along_axis(a, ai, axis=1)
    array([[30],
           [60]])
    """
    a: npt.NDArray = np.asanyarray(a)
    indices: npt.NDArray = np.asanyarray(indices)

    if out is not None:
        out: np.ndarray = np.asanyarray(out)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{indices.dtype=} must be subdtype of np.integer")

    if (axis is None) and (out is None) and a.flags.c_contiguous:
        _check_indices_1d(indices)
        out = np.empty(indices.shape, dtype=a.dtype)
        _take_along_axis_kernel_1d_contiguous_cached_parallel(a, indices, out)
        return out

    if (axis is None) and (out is None) and (not a.flags.c_contiguous):
        _check_indices_1d(indices)
        out = np.empty(indices.shape, dtype=a.dtype)
        _take_along_axis_kernel_1d_noncontiguous_cached_parallel(a, indices, out)
        return out

    if (axis is None) and isinstance(out, np.ndarray) and a.flags.c_contiguous:
        _check_indices_1d(indices)
        _check_out_shape(indices, out)
        _take_along_axis_kernel_1d_contiguous_cached_parallel(a, indices, out)
        return out

    if (axis is None) and isinstance(out, np.ndarray) and (not a.flags.c_contiguous):
        _check_indices_1d(indices)
        _check_out_shape(indices, out)
        _take_along_axis_kernel_1d_noncontiguous_cached_parallel(a, indices, out)
        return out

    if isinstance(axis, SupportsIndex) and (out is None):
        _check_shapes(a, indices, axis)
        out = np.empty(indices.shape, dtype=a.dtype)
        _take_along_axis_kernel_cached_parallel(a, indices, axis, out)
        return out

    if isinstance(axis, SupportsIndex) and isinstance(out, np.ndarray):
        _check_shapes(a, indices, axis)
        _check_out_shape(indices, out)
        _take_along_axis_kernel_cached_parallel(a, indices, axis, out)
        return out

    raise TypeError(
        f"{type(out).__name__=} must be None or Array. {type(axis).__name__=} must be None or int"
    )


@overload(take_along_axis, inline="always")
def overload_take_along_axis(a, indices, axis=None, out=None):
    if not isinstance(a, nbt.Array) or not isinstance(indices, nbt.Array):
        raise nb.errors.TypingError("`a` and `indices` must be arrays")

    if isinstance(axis, nbt.Optional):
        axis = axis.type
    if isinstance(out, nbt.Optional):
        out = out.type

    if (
        isinstance(axis, OMIT_OR_NONE)
        and isinstance(out, OMIT_OR_NONE)
        and (a.layout == "C")
    ):

        def impl(a, indices, axis=None, out=None):
            _check_indices_1d(indices)
            result = np.empty(indices.shape, dtype=a.dtype)
            _take_along_axis_kernel_1d_contiguous_inline(a, indices, result)
            return result

        return impl

    if (
        isinstance(axis, OMIT_OR_NONE)
        and isinstance(out, OMIT_OR_NONE)
        and (a.layout != "C")
    ):

        def impl(a, indices, axis=None, out=None):
            _check_indices_1d(indices)
            result = np.empty(indices.shape, dtype=a.dtype)
            _take_along_axis_kernel_1d_noncontiguous_inline(a, indices, result)
            return result

        return impl

    if (
        isinstance(axis, OMIT_OR_NONE)
        and isinstance(out, nbt.Array)
        and (a.layout == "C")
    ):

        def impl(a, indices, axis=None, out=None):
            _check_indices_1d(indices)
            _check_out_shape(indices, out)
            _take_along_axis_kernel_1d_contiguous_inline(a, indices, out)
            return out

        return impl

    if (
        isinstance(axis, OMIT_OR_NONE)
        and isinstance(out, nbt.Array)
        and (a.layout != "C")
    ):

        def impl(a, indices, axis=None, out=None):
            _check_indices_1d(indices)
            _check_out_shape(indices, out)
            _take_along_axis_kernel_1d_noncontiguous_inline(a, indices, out)
            return out

        return impl

    if isinstance(axis, nbt.Integer) and isinstance(out, OMIT_OR_NONE):

        def impl(a, indices, axis=None, out=None):
            _check_shapes(a, indices, axis)
            result = np.empty(indices.shape, dtype=a.dtype)
            _take_along_axis_kernel_inline(a, indices, axis, result)
            return result

        return impl

    if isinstance(axis, nbt.Integer) and isinstance(out, nbt.Array):

        def impl(a, indices, axis=None, out=None):
            _check_shapes(a, indices, axis)
            _check_out_shape(indices, out)
            _take_along_axis_kernel_inline(a, indices, axis, out)
            return out

        return impl

    raise TypeError(
        f"{type(out)=} must be None or Array. {type(axis)=} must be None or int"
    )
