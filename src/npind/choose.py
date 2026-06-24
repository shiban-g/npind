from typing import Literal, Optional, Sequence, Union

import numba as nb
import numpy as np
from numba import types as nbt
from numba.extending import overload, register_jitable
from numpy import typing as npt

from .utils import (
    OMIT_OR_NONE,
    check_clipmode,
    clip_if_outside,
    may_share_memory,
    raise_if_outside,
    wrap_if_outside,
)


@register_jitable(nopython=True)
def broadcast_shape(
    a: npt.NDArray, choices: tuple[npt.NDArray, ...]
) -> tuple[int, ...]:
    shape = a.shape
    return _broadcast_shape_helper(shape, choices)


@register_jitable(nopython=True)
def _broadcast_shape_helper(shape, choices):
    shape = np.broadcast_shapes(shape, choices[0].shape)
    if len(choices) == 1:
        return shape
    return _broadcast_shape_helper(shape, choices[1:])


@register_jitable(nopython=True)
def broadcast_arrays(
    a: npt.NDArray, choices: tuple[npt.NDArray, ...]
) -> tuple[npt.NDArray, tuple[npt.NDArray, ...]]:
    # The output of `np.broadcast_arrays` is a list.
    # To improve performance, do not use `np.broadcast_arrays`.
    shape = broadcast_shape(a, choices)

    # Recursive functions are used to handle heterogeneous tuples.
    choices = _broadcast_arrays_helper(choices, shape)
    a = np.broadcast_to(a, shape)
    return a, choices


@register_jitable(nopython=True)
def _broadcast_arrays_helper(
    arrays: tuple[npt.NDArray, ...], shape: tuple[int, ...]
) -> tuple[npt.NDArray, ...]:
    head = (np.broadcast_to(arrays[0], shape),)
    if len(arrays) == 1:
        return head

    tail = _broadcast_arrays_helper(arrays[1:], shape)
    return head + tail


@register_jitable(nopython=True)
def _may_share_memory_choose(a, choices, out):
    result = may_share_memory(a, out)
    return result or _may_share_memory_chelper(choices, out)


@register_jitable(nopython=True)
def _may_share_memory_chelper(choices, out):
    result = may_share_memory(choices[0], out)
    if len(choices) == 1:
        return result
    return result or _may_share_memory_chelper(choices[1:], out)


def _choose_kernel_array(a, choices, out, mode):
    # If type of `choices` is numba.types.Array

    check_clipmode(mode)
    n_choices = len(choices)
    shape = np.broadcast_shapes((1,) + a.shape, choices.shape)
    a = np.broadcast_to(a, shape[1:])
    choices = np.broadcast_to(choices, shape)

    out_buf = out
    if may_share_memory(a, out) or may_share_memory(a, choices):
        out_buf = np.empty_like(out)

    if mode == "raise":
        for idx in nb.pndindex(a.shape):
            i = raise_if_outside(a[idx], n_choices)
            out_buf[idx] = choices[(i,) + idx]

    if mode == "wrap":
        for idx in nb.pndindex(a.shape):
            i = wrap_if_outside(a[idx], n_choices)
            out_buf[idx] = choices[(i,) + idx]

    if mode == "clip":
        for idx in nb.pndindex(a.shape):
            i = clip_if_outside(a[idx], n_choices)
            out_buf[idx] = choices[(i,) + idx]

    if out is not out_buf:
        out[...] = out_buf


def _choose_kernel_homo(a, choices, out, mode):
    # If type of `choices` is numba.types.UniTuple

    check_clipmode(mode)
    n_choices = len(choices)
    a, choices = broadcast_arrays(a, choices)

    out_buf = out
    if _may_share_memory_choose(a, choices, out):
        out_buf = np.empty_like(out)

    if mode == "raise":
        for idx in nb.pndindex(a.shape):
            i = raise_if_outside(a[idx], n_choices)
            out_buf[idx] = choices[i][idx]

    if mode == "wrap":
        for idx in nb.pndindex(a.shape):
            i = wrap_if_outside(a[idx], n_choices)
            out_buf[idx] = choices[i][idx]

    if mode == "clip":
        for idx in nb.pndindex(a.shape):
            i = clip_if_outside(a[idx], n_choices)
            out_buf[idx] = choices[i][idx]

    if out is not out_buf:
        out[...] = out_buf


@nb.njit
def _hetero_loop_inside(choices, out_buf, tup_idx, arr_idx):
    # Hack. I wanted to use `literal_unroll`, but I couldn't resolve the error.

    # i = 0
    # for choice in nb.literal_unroll(choices):
    #     if i == tup_idx:
    #         out_buf[arr_idx] = choice[arr_idx]
    #     i += 1
    if len(choices) == 0:
        return
    if tup_idx == 0:
        out_buf[arr_idx] = choices[0][arr_idx]
        return
    _hetero_loop_inside(choices[1:], out_buf, tup_idx - 1, arr_idx)


def _choose_kernel_hetero(a, choices, out, mode):
    # If type of `choices` is numba.types.Tuple

    check_clipmode(mode)
    n_choices = len(choices)
    a, choices = broadcast_arrays(a, choices)

    out_buf = out
    if _may_share_memory_choose(a, choices, out):
        out_buf = np.empty_like(out)

    if mode == "raise":
        for idx in nb.pndindex(a.shape):
            i = raise_if_outside(a[idx], n_choices)
            _hetero_loop_inside(choices, out_buf, i, idx)

    if mode == "wrap":
        for idx in nb.pndindex(a.shape):
            i = wrap_if_outside(a[idx], n_choices)
            _hetero_loop_inside(choices, out_buf, i, idx)

    if mode == "clip":
        for idx in nb.pndindex(a.shape):
            i = clip_if_outside(a[idx], n_choices)
            _hetero_loop_inside(choices, out_buf, i, idx)

    if out is not out_buf:
        out[...] = out_buf


_choose_kernel_array_cached_parallel = nb.njit(
    _choose_kernel_array, boundscheck=False, cache=True, parallel=True
)
_choose_kernel_homo_cached_parallel = nb.njit(
    _choose_kernel_homo, boundscheck=False, cache=True, parallel=True
)
_choose_kernel_hetero_cached_parallel = nb.njit(
    _choose_kernel_hetero, boundscheck=False, cache=True, parallel=True
)
_choose_kernel_array_inline = nb.njit(_choose_kernel_array, inline="always")
_choose_kernel_homo_inline = nb.njit(_choose_kernel_homo, inline="always")
_choose_kernel_hetero_inline = nb.njit(_choose_kernel_hetero, inline="always")


@register_jitable(nopython=True)
def check_out_shape(array, shape):
    if array.shape != shape:
        raise ValueError("shape of out-array does not match result of choose")


def as_any_arrays(arrays: Sequence[npt.ArrayLike]) -> tuple[npt.NDArray, ...]:
    return tuple(np.asanyarray(arr) for arr in arrays)


def have_same_dtypes(arrays: tuple[npt.NDArray]) -> bool:
    dtype = arrays[0].dtype
    return all(arr.dtype == dtype for arr in arrays)


def choose(
    a: npt.ArrayLike,
    choices: Union[npt.ArrayLike, Sequence[npt.ArrayLike]],
    out: Optional[npt.NDArray] = None,
    mode: Literal["raise", "wrap", "clip"] = "raise",
) -> npt.NDArray:
    a = np.asanyarray(a)
    if not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"{a.dtype=} must be subdtype of np.integer")

    if isinstance(choices, np.ndarray) and isinstance(out, np.ndarray):
        shape = np.broadcast_shapes(a.shape, choices.shape[1:])
        check_out_shape(out, shape)
        _choose_kernel_array_cached_parallel(a, choices, out, mode)
        return out

    if isinstance(choices, np.ndarray) and (out is None):
        shape = np.broadcast_shapes(a.shape, choices.shape[1:])
        out = np.empty(shape, dtype=choices.dtype)
        _choose_kernel_array_cached_parallel(a, choices, out, mode)
        return out

    if isinstance(choices, Sequence) and isinstance(out, np.ndarray):
        choices = as_any_arrays(choices)
        shape = np.broadcast_shapes(a.shape, *[c.shape for c in choices])
        check_out_shape(out, shape)
        if have_same_dtypes(choices):
            _choose_kernel_homo_cached_parallel(a, choices, out, mode)
        else:
            _choose_kernel_hetero_cached_parallel(a, choices, out, mode)
        return out

    if isinstance(choices, Sequence) and (out is None):
        choices = as_any_arrays(choices)
        shape = np.broadcast_shapes(a.shape, *[c.shape for c in choices])
        dtype = np.result_type(*[c.dtype for c in choices])
        out = np.empty(shape, dtype=dtype)
        if have_same_dtypes(choices):
            _choose_kernel_homo_cached_parallel(a, choices, out, mode)
        else:
            _choose_kernel_hetero_cached_parallel(a, choices, out, mode)
        return out

    raise TypeError()


def prepare_nb_dtype(choices):
    np_dtypes = [np.dtype(c.dtype.name) for c in choices]
    np_dtype = np.result_type(*np_dtypes)
    nb_dtype = nb.from_dtype(np_dtype)
    return nb_dtype


@overload(choose, inline="always")
def overload_choose(a, choices, out=None, mode="raise"):
    if isinstance(out, nbt.Optional):
        out = out.type

    if not isinstance(a, nbt.Array):
        raise TypeError("`a` must be an array")

    if isinstance(choices, (nbt.Tuple, nbt.UniTuple)) and (
        not all(isinstance(t, nbt.Array) for t in choices.types)
    ):
        raise TypeError()
    if isinstance(choices, (nbt.List, nbt.ListType)):
        raise NotImplementedError()

    if isinstance(choices, nbt.Array) and isinstance(out, nbt.Array):

        def impl(a, choices, out=None, mode="raise"):
            shape = np.broadcast_shapes((1,) + a.shape, choices.shape)
            check_out_shape(out, shape[1:])
            _choose_kernel_array_inline(a, choices, out, mode)
            return out

        return impl

    if isinstance(choices, nbt.Array) and isinstance(out, OMIT_OR_NONE):

        def impl(a, choices, out=None, mode="raise"):
            shape = np.broadcast_shapes((1,) + a.shape, choices.shape)
            out = np.empty(shape[1:], dtype=choices.dtype)
            _choose_kernel_array_inline(a, choices, out, mode)
            return out

        return impl

    if isinstance(choices, nbt.UniTuple) and isinstance(out, nbt.Array):

        def impl(a, choices, out=None, mode="raise"):
            shape = broadcast_shape(a, choices)
            check_out_shape(out, shape)
            _choose_kernel_homo_inline(a, choices, out, mode)
            return out

        return impl

    if isinstance(choices, nbt.UniTuple) and isinstance(out, OMIT_OR_NONE):

        def impl(a, choices, out=None, mode="raise"):
            shape = broadcast_shape(a, choices)
            out = np.empty(shape, dtype=choices[0].dtype)
            _choose_kernel_homo_inline(a, choices, out, mode)
            return out

        return impl

    if isinstance(choices, nbt.Tuple) and isinstance(out, nbt.Array):

        def impl(a, choices, out=None, mode="raise"):
            shape = broadcast_shape(a, choices)
            check_out_shape(out, shape)
            _choose_kernel_hetero_inline(a, choices, out, mode)
            return out

        return impl

    if isinstance(choices, nbt.Tuple) and isinstance(out, OMIT_OR_NONE):
        dtype = prepare_nb_dtype(choices)

        def impl(a, choices, out=None, mode="raise"):
            shape = broadcast_shape(a, choices)
            out = np.empty(shape, dtype=dtype)
            _choose_kernel_hetero_inline(a, choices, out, mode)
            return out

        return impl

    raise TypeError()
