import numba as nb
import numpy as np
import pytest

import npind as npi


class TestChoose:
    """Tests for choose, based on NumPy's documentation examples."""

    def test_basic_1d(self):
        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        a = np.array([2, 3, 1, 0])

        expected = np.choose(a, choices)
        result = npi.choose(a, choices)

        np.testing.assert_array_equal(result, expected)

    def test_mode_clip(self):
        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        a = np.array([2, 4, 1, 0])

        expected = np.choose(a, choices, mode="clip")
        result = npi.choose(a, choices, mode="clip")

        np.testing.assert_array_equal(result, expected)

    def test_mode_wrap(self):
        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        a = np.array([2, 4, 1, 0])

        expected = np.choose(a, choices, mode="wrap")
        result = npi.choose(a, choices, mode="wrap")

        np.testing.assert_array_equal(result, expected)

    def test_broadcast_2d(self):
        a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        choices = [-10, 10]

        expected = np.choose(a, choices)
        result = npi.choose(a, choices)

        np.testing.assert_array_equal(result, expected)

    def test_broadcast_3d(self):
        a = np.array([0, 1]).reshape((2, 1, 1))
        c1 = np.array([1, 2, 3]).reshape((1, 3, 1))
        c2 = np.array([-1, -2, -3, -4, -5]).reshape((1, 1, 5))

        expected = np.choose(a, (c1, c2))
        result = npi.choose(a, (c1, c2))

        np.testing.assert_array_equal(result, expected)

    def test_choices_as_ndarray(self):
        choices = np.array([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
        a = np.array([2, 3, 1, 0])

        expected = np.choose(a, choices)
        result = npi.choose(a, choices)

        np.testing.assert_array_equal(result, expected)


class TestChooseEdgeCases:
    def test_out_provided(self):
        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        a = np.array([2, 3, 1, 0])
        out = np.empty(4, dtype=int)

        expected = np.choose(a, choices)
        result = npi.choose(a, choices, out=out)

        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(out, expected)
        assert result is out

    def test_out_is_a(self):
        a = np.array([2, 3, 1, 0])
        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        expected = np.choose(a, choices)

        result = npi.choose(a, choices, out=a)

        np.testing.assert_array_equal(a, expected)
        np.testing.assert_array_equal(result, expected)
        assert result is a

    def test_non_contiguous_out(self):
        a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        choices = [-10, 10]

        full_out = np.zeros((3, 6), dtype=int)
        target_out = full_out[:, ::2]

        npi.choose(a, choices, out=target_out)

        expected = np.choose(a, choices)
        np.testing.assert_array_equal(target_out, expected)

    def test_out_of_bounds_raise(self):
        a = np.array([0, 4, 1])
        choices = [-10, 10]

        with pytest.raises(IndexError):
            npi.choose(a, choices, mode="raise")

    def test_shape_mismatch(self):
        a = np.arange(6).reshape(2, 3)
        choices = [np.arange(4), np.arange(4)]

        with pytest.raises(ValueError, match="broadcast"):
            npi.choose(a, choices)

    def test_empty_choices(self):
        a = np.array([0, 1, 0])
        choices = []

        with pytest.raises(ValueError, match="at least one array"):
            npi.choose(a, choices)

    def test_non_integer_a(self):
        a = np.array([0.0, 1.0, 0.0])
        choices = [-10, 10]

        with pytest.raises(TypeError, match="integer"):
            npi.choose(a, choices)

    def test_out_shape_mismatch(self):
        a = np.array([0, 1, 0])
        choices = [-10, 10]
        out = np.empty(5, dtype=int)

        with pytest.raises(ValueError, match="shape of out-array"):
            npi.choose(a, choices, out=out)


class TestChooseInJit:
    def test_jit_basic(self):
        @nb.njit
        def jit_func(a, choices):
            return npi.choose(a, choices)

        a = np.array([2, 3, 1, 0])
        choices = (
            np.array([0, 1, 2, 3]),
            np.array([10, 11, 12, 13]),
            np.array([20, 21, 22, 23]),
            np.array([30, 31, 32, 33]),
        )
        expected = np.choose(a, choices)

        result = jit_func(a, choices)
        np.testing.assert_array_equal(result, expected)

    def test_jit_with_out(self):
        @nb.njit
        def jit_func(a, choices, out):
            return npi.choose(a, choices, out=out)

        a = np.array([2, 3, 1, 0])
        choices = (
            np.array([0, 1, 2, 3]),
            np.array([10, 11, 12, 13]),
            np.array([20, 21, 22, 23]),
            np.array([30, 31, 32, 33]),
        )
        out = np.empty(4, dtype=np.int64)
        expected = np.choose(a, choices)

        result = jit_func(a, choices, out)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(out, expected)

    def test_jit_parallel(self):
        @nb.njit(parallel=True)
        def jit_func(a, choices):
            return npi.choose(a, choices)

        a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        choices = (np.array(-10), np.array(10))
        expected = np.choose(a, choices)

        result = jit_func(a, choices)
        np.testing.assert_array_equal(result, expected)

    def test_jit_choices_as_array(self):
        @nb.njit
        def jit_func(a, choices):
            return npi.choose(a, choices)

        a = np.array([2, 3, 1, 0])
        choices = np.array([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
        expected = np.choose(a, choices)

        result = jit_func(a, choices)
        np.testing.assert_array_equal(result, expected)

    def test_jit_out_of_bounds(self):
        @nb.njit
        def jit_func(a, choices):
            return npi.choose(a, choices)

        a = np.array([0, 4, 1])
        choices = (np.array(-10), np.array(10))

        with pytest.raises(IndexError):
            jit_func(a, choices)
