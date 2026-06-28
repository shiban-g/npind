from typing import Literal, Optional, Sequence, Union

import numba as nb
import numpy as np
from numba import literal_unroll
from numba import types as nbt
from numba.extending import overload
from numpy import typing as npt

from .utils import (
    OMIT_OR_NONE,
    broadcast_arrays,
    broadcast_shapes,
    check_clipmode,
    check_out_shape,
    clip_if_outside,
    may_share_memories,
    raise_if_outside,
    wrap_if_outside,
)


def _create_out_array(
    a: npt.NDArray,
    choices: Union[tuple[npt.NDArray, ...], npt.NDArray],
    out: Optional[npt.NDArray],
):
    if isinstance(choices, np.ndarray) and isinstance(out, np.ndarray):
        shape = np.broadcast_shapes((1,) + a.shape, choices.shape)
        check_out_shape(out, shape[1:])
        return out

    if isinstance(choices, np.ndarray) and (out is None):
        shape = np.broadcast_shapes((1,) + a.shape, choices.shape)
        return np.empty(shape[1:], dtype=choices.dtype)

    if isinstance(choices, tuple) and isinstance(out, np.ndarray):
        shape = broadcast_shapes((a,) + choices)
        check_out_shape(out, shape)
        return out

    if isinstance(choices, tuple) and (out is None):
        dtype = np.result_type(*[c.dtype for c in choices])
        shape = broadcast_shapes((a,) + choices)
        return np.empty(shape, dtype=dtype)

    raise TypeError()


@overload(_create_out_array)
def overload_create_out_array(a, choices, out):
    if isinstance(choices, nbt.Array) and isinstance(out, nbt.Array):

        def impl(a, choices, out):
            shape = np.broadcast_shapes((1,) + a.shape, choices.shape)
            check_out_shape(out, shape[1:])
            return out

        return impl

    if isinstance(choices, nbt.Array) and isinstance(out, OMIT_OR_NONE):

        def impl(a, choices, out):
            shape = np.broadcast_shapes((1,) + a.shape, choices.shape)
            return np.empty(shape[1:], dtype=choices.dtype)

        return impl

    if isinstance(choices, nbt.BaseTuple) and isinstance(out, nbt.Array):

        def impl(a, choices, out):
            shape = broadcast_shapes((a,) + choices)
            check_out_shape(out, shape)
            return out

        return impl

    if isinstance(choices, nbt.BaseTuple) and isinstance(out, OMIT_OR_NONE):
        np_dtypes = [np.dtype(c.dtype.name) for c in choices]
        np_dtype = np.result_type(*np_dtypes)
        nb_dtype = nb.from_dtype(np_dtype)

        def impl(a, choices, out):
            shape = broadcast_shapes((a,) + choices)
            return np.empty(shape, dtype=nb_dtype)

        return impl
    raise TypeError()


def _broadcast_arrays(
    a: npt.NDArray, choices: Union[tuple[npt.NDArray, ...], npt.NDArray]
):
    raise NotImplementedError()


@overload(_broadcast_arrays)
def overload_broadcast_arrays(a, choices):
    if isinstance(choices, nbt.Array):

        def impl(a, choices):
            shape = np.broadcast_shapes((1,) + a.shape, choices.shape)
            a = np.broadcast_to(a, shape[1:])
            choices = np.broadcast_to(choices, shape)
            return a, choices

        return impl

    if isinstance(choices, nbt.BaseTuple):

        def impl(a, choices):
            shape = broadcast_shapes((a,) + choices)
            choices = broadcast_arrays(choices, shape)
            a = np.broadcast_to(a, shape)
            return a, choices

        return impl
    raise TypeError()


def _may_share_memory(
    a: npt.NDArray,
    choices: Union[tuple[npt.NDArray, ...], npt.NDArray],
    out: npt.NDArray,
):
    # Here is unreachable.
    # this method is required to define `overload_may_share_memory`.
    raise NotImplementedError()


@overload(_may_share_memory)
def overload_may_share_memory(a, choices, out):
    if isinstance(choices, nbt.Array):

        def impl(a, choices, out):
            return may_share_memories((a, choices), out)

        return impl

    if isinstance(choices, nbt.BaseTuple):

        def impl(a, choices, out):
            return may_share_memories((a,) + choices, out)

        return impl
    raise TypeError()


def _loop_inside(
    choices: Union[tuple[npt.NDArray, ...], npt.NDArray],
    out_buf: npt.NDArray,
    tup_idx: int,
    arr_idx: tuple[int, ...],
):
    # Here is unreachable.
    # this method is required to define `overload_loop_inside`.
    raise NotImplementedError()


# `@nb.njit(inline="always")` cause an error.
# `inline="always"` is required for performance reasons.
@overload(_loop_inside, inline="always")
def overload_loop_inside(choices, out_buf, tup_idx, arr_idx):

    if isinstance(choices, nbt.Array):

        def impl(choices, out_buf, tup_idx, arr_idx):
            out_buf[arr_idx] = choices[(tup_idx,) + arr_idx]

        return impl

    if isinstance(choices, nbt.UniTuple):

        def impl(choices, out_buf, tup_idx, arr_idx):
            out_buf[arr_idx] = choices[tup_idx][arr_idx]

        return impl

    if isinstance(choices, nbt.Tuple):
        indices = tuple(range(choices.count))

        def impl(choices, out_buf, tup_idx, arr_idx):
            # Adding `nb.` causes an error. The compiler might fail to parse the syntax.
            # for i in nb.literal_unroll(indices):
            for i in literal_unroll(indices):
                if i == tup_idx:
                    out_buf[arr_idx] = choices[i][arr_idx]

        return impl
    raise TypeError()


def _choose_kernel(a, choices, out, mode):
    check_clipmode(mode)
    a, choices = _broadcast_arrays(a, choices)

    n_choices = len(choices)

    out_buf = out
    if _may_share_memory(a, choices, out):
        out_buf = np.empty_like(out)

    if mode == "raise":
        raise_if_outside(a, n_choices)
        for idx in nb.pndindex(a.shape):
            _loop_inside(choices, out_buf, a[idx], idx)

    if mode == "wrap":
        for idx in nb.pndindex(a.shape):
            i = wrap_if_outside(a[idx], n_choices)
            _loop_inside(choices, out_buf, i, idx)

    if mode == "clip":
        for idx in nb.pndindex(a.shape):
            i = clip_if_outside(a[idx], n_choices)
            _loop_inside(choices, out_buf, i, idx)

    if out is not out_buf:
        out[...] = out_buf


_choose_kernel_cached_parallel = nb.njit(
    _choose_kernel, boundscheck=False, cache=True, parallel=True
)
_choose_kernel_inline = nb.njit(_choose_kernel, inline="always")


def choose(
    a: npt.ArrayLike,
    choices: Union[npt.ArrayLike, Sequence[npt.ArrayLike]],
    out: Optional[npt.NDArray] = None,
    mode: Literal["raise", "wrap", "clip"] = "raise",
) -> npt.NDArray:
    a = np.asanyarray(a)
    if not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"{a.dtype=} must be subdtype of np.integer")

    if isinstance(choices, Sequence):
        if len(choices) == 0:
            raise ValueError("at least one array")
        if all(isinstance(c, np.ndarray) for c in choices):
            choices: tuple[npt.NDArray, ...] = tuple(choices)
        else:
            choices: npt.NDArray = np.asanyarray(choices)

            # To make `choices` broadcastable
            choices = choices.reshape(
                choices.shape[:1]
                + (1,) * (a.ndim - choices.ndim + 1)
                + choices.shape[1:]
            )

    if isinstance(choices, (np.ndarray, tuple)) and (
        isinstance(out, np.ndarray) or (out is None)
    ):
        out = _create_out_array(a, choices, out)
        _choose_kernel_cached_parallel(a, choices, out, mode)
        return out

    raise TypeError()


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

    if isinstance(choices, (nbt.BaseTuple, nbt.Array)) and (
        isinstance(out, (nbt.Array,) + OMIT_OR_NONE)
    ):

        def impl(a, choices, out=None, mode="raise"):
            out_ = _create_out_array(a, choices, out)
            _choose_kernel_inline(a, choices, out_, mode)
            return out_

        return impl

    raise TypeError()
