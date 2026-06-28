from typing import Literal, Optional, SupportsIndex

import numba as nb
import numpy as np
from numba import types as nbt
from numba.extending import overload
from numpy import typing as npt

from .utils import (
    OMIT_OR_NONE,
    axes_to_left,
    check_clipmode,
    check_out_shape,
    clip_if_outside,
    may_share_memory,
    raise_if_outside,
    replace_item_with_tuple,
    wrap_if_outside,
)


def _create_out_array(
    a: npt.NDArray,
    indices: npt.NDArray,
    axis: Optional[int],
    out: Optional[npt.NDArray],
):
    if axis is None and isinstance(out, np.ndarray):
        check_out_shape(out, indices.shape)
        return out

    if axis is None and (out is None):
        return np.empty(indices.shape, dtype=a.dtype)

    if isinstance(axis, SupportsIndex) and isinstance(out, np.ndarray):
        expected_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
        check_out_shape(out, expected_shape)
        return out

    if isinstance(axis, SupportsIndex) and (out is None):
        result_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
        return np.empty(result_shape, dtype=a.dtype)

    raise TypeError()


@overload(_create_out_array)
def overload_create_out_array(a, indices, axis, out):
    if isinstance(axis, OMIT_OR_NONE) and isinstance(out, nbt.Array):

        def impl(a, indices, axis, out):
            check_out_shape(out, indices.shape)
            return out

        return impl

    if isinstance(axis, OMIT_OR_NONE) and isinstance(out, OMIT_OR_NONE):

        def impl(a, indices, axis, out):
            return np.empty(indices.shape, dtype=a.dtype)

        return impl

    if isinstance(axis, nbt.Integer) and isinstance(out, nbt.Array):

        def impl(a, indices, axis, out):
            expected_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
            check_out_shape(out, expected_shape)
            return out

        return impl

    if isinstance(axis, nbt.Integer) and isinstance(out, OMIT_OR_NONE):

        def impl(a, indices, axis, out):
            result_shape = replace_item_with_tuple(a.shape, indices.shape, axis)
            return np.empty(result_shape, dtype=a.dtype)

        return impl

    raise TypeError()


def _loop_inside(
    a: npt.NDArray,
    out_buf: npt.NDArray,
    src_idx: int,
    arr_idx: tuple[int, ...],
    axis: Optional[int] = None,
):
    # Here is unreachable.
    # this method is required to define `overload_loop_inside`.
    raise NotImplementedError()


# `@nb.njit(inline="always")` cause an error.
# `inline="always"` is required for performance reasons.
@overload(_loop_inside, inline="always")
def overload_loop_inside(a, out_buf, src_idx, arr_idx, axis=None):

    if isinstance(axis, OMIT_OR_NONE) and (a.layout == "C"):

        def impl(a, out_buf, src_idx, arr_idx, axis=None):
            out_buf[arr_idx] = a[src_idx]

        return impl

    if isinstance(axis, OMIT_OR_NONE) and (a.layout != "C"):

        def impl(a, out_buf, src_idx, arr_idx, axis=None):
            out_buf[arr_idx] = a.flat[src_idx]

        return impl

    if isinstance(axis, nbt.Integer):

        def impl(a, out_buf, src_idx, arr_idx, axis=None):
            out_buf[arr_idx] = a[src_idx]

        return impl

    raise NotImplementedError()


def _prepare(
    a: npt.NDArray, indices: npt.NDArray, axis: Optional[int], out: npt.NDArray
):
    raise NotImplementedError()


@overload(_prepare)
def overload_prepare(a, indices, axis, out):
    if isinstance(axis, OMIT_OR_NONE) and (a.layout == "C"):

        def impl(a, indices, axis, out):
            bound = a.size
            a = a.ravel()
            return a, out, bound

        return impl

    if isinstance(axis, OMIT_OR_NONE) and (a.layout != "C"):

        def impl(a, indices, axis, out):
            bound = a.size
            return a, out, bound

        return impl

    if isinstance(axis, nbt.Integer):

        def impl(a, indices, axis, out):
            a = axes_to_left(a, axis, axis + 1)
            out = axes_to_left(out, axis, axis + indices.ndim)
            bound = len(a)
            return a, out, bound

        return impl

    raise NotImplementedError()


def _take_kernel(a, indices, axis, out, mode):
    check_clipmode(mode)

    b, out, bound = _prepare(a, indices, axis, out)

    out_buf = out
    if may_share_memory(b, out) or may_share_memory(indices, out):
        out_buf = np.empty_like(out)

    if mode == "raise":
        raise_if_outside(indices, bound)
        for idx in nb.pndindex(indices.shape):
            _loop_inside(b, out_buf, indices[idx], idx, axis)

    if mode == "wrap":
        for idx in nb.pndindex(indices.shape):
            i = wrap_if_outside(indices[idx], bound)
            _loop_inside(b, out_buf, i, idx, axis)

    if mode == "clip":
        for idx in nb.pndindex(indices.shape):
            i = clip_if_outside(indices[idx], bound)
            _loop_inside(b, out_buf, i, idx, axis)

    if out is not out_buf:
        out[...] = out_buf


_take_kernel_inline = nb.njit(_take_kernel, boundscheck=False, inline="always")
_take_kernel_cached_parallel = nb.njit(
    _take_kernel, boundscheck=False, cache=True, parallel=True
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

    if (isinstance(out, np.ndarray) or out is None) and (
        axis is None or isinstance(axis, SupportsIndex)
    ):
        out = _create_out_array(a, indices, axis, out)
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

    if isinstance(axis, (nbt.Integer,) + OMIT_OR_NONE) and isinstance(
        out, (nbt.Array,) + OMIT_OR_NONE
    ):

        def impl(a, indices, axis=None, out=None, mode="raise"):
            out_ = _create_out_array(a, indices, axis, out)
            _take_kernel_inline(a, indices, axis, out_, mode)
            return out_

        return impl
    raise TypeError(
        f"{type(out)=} must be None or Array. {type(axis)=} must be None or int"
    )
