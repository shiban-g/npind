from typing import Literal, Optional

import numba as nb
import numpy as np
import numpy.typing as npt


@nb.njit(parallel=True, cache=True, boundscheck=False)
def _take_kernel(a, indices, axis, out):
    a = np.moveaxis(a, axis, 0)

    for i in range(indices.ndim):
        out = np.moveaxis(out, axis + i, i)

    for idx in nb.pndindex(indices.shape):
        out[idx] = a[indices[idx]]

    for i in range(indices.ndim - 1, -1, -1):
        out = np.moveaxis(out, i, axis + i)


@nb.njit(parallel=True, cache=True, boundscheck=False)
def _take_kernel_none_discontinuous(a, indices, out):
    for idx in nb.pndindex(indices.shape):
        out[idx] = a.flat[indices[idx]]


@nb.njit(parallel=True, cache=True, boundscheck=False)
def _take_kernel_none_continuous(a, indices, out):
    b = a.ravel()
    for idx in nb.pndindex(indices.shape):
        out[idx] = b[indices[idx]]


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

    length = a.size if axis is None else a.shape[axis]
    if (mode == "raise") and (indices.size > 0):
        min_ = indices.min()
        max_ = indices.max()
        if not (0 <= min_ <= max_ < length):
            raise IndexError("index out of bounds")

    if mode == "warp":
        indices = indices % length

    if mode == "clip":
        indices = indices.clip(0, length) % length

    result_shape = indices.shape
    if axis is not None:
        result_shape = a.shape[:axis] + indices.shape + a.shape[axis + 1 :]

    if out is None:
        out = np.empty(result_shape, dtype=a.dtype)
    out: np.ndarray = np.asanyarray(out)

    if out.shape != result_shape:
        raise ValueError("Output shape mismatch")

    out_buf = out
    if np.shares_memory(a, out_buf) or np.shares_memory(indices, out_buf):
        out_buf = np.empty_like(out)

    if axis is None:
        if a.flags.c_contiguous:
            _take_kernel_none_continuous(a, indices, out_buf)
        else:
            _take_kernel_none_discontinuous(a, indices, out_buf)
    else:
        _take_kernel(a, indices, axis, out_buf)

    if np.shares_memory(out, out_buf):
        return out
    np.copyto(out, out_buf)
    return out
