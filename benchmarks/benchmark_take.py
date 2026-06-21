import pathlib
import time

import numba as nb
import numpy as np
import numpy.typing as npt

import npind as npi


def get_contiguous_int_axis():
    rnd_gen = np.random.default_rng(0)
    a = rnd_gen.random((1000, 1000))
    indeces = rnd_gen.integers(1000, size=1000)

    axis = 0
    out = np.empty((1000, 1000))

    return a, indeces, axis, out


def get_discontiguous_int_axis():
    rnd_gen = np.random.default_rng(0)
    a = rnd_gen.random((1000, 1001))
    indeces = rnd_gen.integers(1000, size=2000)
    a = a[:, 1:].T
    indeces = indeces[::2]

    axis = 0
    out = np.empty((1000, 1000))

    return a, indeces, axis, out


def get_contiguous_none_axis():
    rnd_gen = np.random.default_rng(0)
    a = rnd_gen.random((1000, 1000))
    indeces = rnd_gen.integers(1000000, size=1000000)

    axis = None
    out = np.empty(1000000)

    return a, indeces, axis, out


def get_discontiguous_none_axis():
    rnd_gen = np.random.default_rng(0)
    a = rnd_gen.random((1000, 1001))
    indeces = rnd_gen.integers(1000000, size=2000000)
    a = a[:, 1:].T
    indeces = indeces[::2]

    axis = None
    out = np.empty(1000000)

    return a, indeces, axis, out


@nb.njit(boundscheck=False)
def nb_take(a, indeces, axis):
    return np.take(a, indeces, axis)


def benchmark(a: npt.NDArray, indeces: npt.NDArray, axis, out: npt.NDArray):
    n_trials = 1000

    nb_take(a, indeces, axis=axis)
    npi.take(a, indeces, axis=axis)
    npi.take(a, indeces, out=out, axis=axis)

    results = {}

    start = time.perf_counter()
    for _ in range(n_trials):
        np.take(a, indeces, axis=axis)
    proc_time = time.perf_counter() - start
    results["numpy_outplace"] = proc_time / n_trials

    start = time.perf_counter()
    for _ in range(n_trials):
        np.take(a, indeces, axis=axis, out=out)
    proc_time = time.perf_counter() - start
    results["numpy_inplace"] = proc_time / n_trials

    start = time.perf_counter()
    for _ in range(n_trials):
        nb_take(a, indeces, axis=axis)
    proc_time = time.perf_counter() - start
    results["numba_outplace"] = proc_time / n_trials

    start = time.perf_counter()
    for _ in range(n_trials):
        npi.take(a, indeces, axis=axis)
    proc_time = time.perf_counter() - start
    results["npind_outplace"] = proc_time / n_trials

    start = time.perf_counter()
    for _ in range(n_trials):
        npi.take(a, indeces, axis=axis, out=out)
    proc_time = time.perf_counter() - start
    results["npind_inplace"] = proc_time / n_trials

    return results


def show_result(header, results):
    print(header)
    for k, v in results.items():
        print(f"{k:25} : {v * 1e3:.3f} [ms]")


def write_result(filename, results):
    with open(filename, "w") as f:
        for k, v in results.items():
            f.write(f"{k} {v * 1e3}\n")


if __name__ == "__main__":
    result_dir = pathlib.Path(__file__).parent / "results"
    result_dir.mkdir(exist_ok=True)

    args = get_contiguous_none_axis()
    results = benchmark(*args)

    show_result("take contiguous data to 1d array", results)
    write_result(result_dir / "contiguous_data_to_1d_arrays.txt", results)

    args = get_contiguous_int_axis()
    results = benchmark(*args)

    show_result("take contiguous data to Nd array", results)
    write_result(result_dir / "contiguous_data_to_Nd_arrays.txt", results)

    args = get_discontiguous_none_axis()
    results = benchmark(*args)

    show_result("take discontiguous data to 1d array", results)
    write_result(result_dir / "discontiguous_data_to_1d_arrays.txt", results)

    args = get_discontiguous_int_axis()
    results = benchmark(*args)

    show_result("take discontiguous data to Nd array", results)
    write_result(result_dir / "discontiguous_data_to_Nd_arrays.txt", results)
