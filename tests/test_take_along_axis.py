import numba as nb
import numpy as np
import pytest

import npind as npi


class TestTakeAlongAxis:
    """
    Tests for take_along_axis, based on NumPy's documentation examples.
    """

    def test_take_along_axis(self):

        a = np.array([0, 1, 2])
        indices = np.array([2, 0, 1])
        # Ground truth from NumPy
        expected = np.take_along_axis(a, indices, axis=0)

        # Result from npind
        result = npi.take_along_axis(a, indices, axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_sort_along_axis(self):
        """
        Replicates the sorting example from the numpy.take_along_axis documentation.
        This tests the basic functionality of reordering elements along an axis.
        """
        a = np.array([[10, 30, 20], [60, 40, 50]])
        # Get the indices that would sort the array along axis 1
        indices = np.argsort(a, axis=1)

        # Ground truth from NumPy
        expected = np.take_along_axis(a, indices, axis=1)

        # Result from npind
        result = npi.take_along_axis(a, indices, axis=1)

        np.testing.assert_array_equal(result, expected)
        # Expected result:
        # array([[10, 20, 30],
        #        [40, 50, 60]])

    def test_get_max_along_axis(self):
        """
        Replicates the argmax example from the numpy.take_along_axis documentation.
        This tests broadcasting and using a smaller index array.
        """
        a = np.array([[10, 30, 20], [60, 40, 50]])
        # Get the index of the maximum value along axis 1
        indices = np.expand_dims(np.argmax(a, axis=1), axis=1)

        # Ground truth from NumPy
        expected = np.take_along_axis(a, indices, axis=1)

        # Result from npind
        result = npi.take_along_axis(a, indices, axis=1)

        np.testing.assert_array_equal(result, expected)
        # Expected result:
        # array([[30],
        #        [60]])

    def test_take_along_axis_axis_none(self):
        """Test flattening behavior when axis is None."""
        a = np.array([[10, 30, 20], [60, 40, 50]])
        indices = np.array([0, 2, 1, 2, 0, 1])

        expected = np.take_along_axis(a, indices, axis=None)
        result = npi.take_along_axis(a, indices, axis=None)

        np.testing.assert_array_equal(result, expected)

    def test_take_along_axis_axis_none_1d(self):
        """Test axis=None with 1D input arrays."""
        a = np.array([0, 1, 2])
        indices = np.array([2, 0, 1])

        expected = np.take_along_axis(a, indices, axis=None)
        result = npi.take_along_axis(a, indices, axis=None)

        np.testing.assert_array_equal(result, expected)

    def test_take_along_axis_axis_none_invalid_indices(self):
        """Test that multi-dimensional indices are rejected when axis is None."""
        a = np.array([[10, 30, 20], [60, 40, 50]])
        indices = np.array([[0, 2, 1], [2, 0, 1]])

        with pytest.raises(ValueError, match="single dimension"):
            npi.take_along_axis(a, indices, axis=None)


class TestTakeAlongAxisEdgeCases:
    def test_out_is_a(self):
        """Case where 'a' and 'out' are the same array (in-place behavior)."""
        a = np.array([10, 20, 30, 40, 50])
        indices = np.array([4, 3, 2, 1, 0])
        expected = np.take_along_axis(a, indices, axis=0)

        result = npi.take_along_axis(a, indices, out=a, axis=0)

        np.testing.assert_array_equal(a, expected)
        np.testing.assert_array_equal(result, expected)
        assert result is a

    def test_out_is_indices(self):
        """Case where 'indices' and 'out' are the same array."""
        a = np.array([10, 20, 30, 40, 50])
        indices = np.array([0, 2, 4, 1, 3], dtype=np.intp)
        expected = np.take_along_axis(a, indices, axis=0)

        result = npi.take_along_axis(a, indices, out=indices, axis=0)

        np.testing.assert_array_equal(result, expected)
        assert result is indices

    def test_all_same_array(self):
        """[Extreme Edge Case] When 'a', 'indices', and 'out' are all the same object."""
        # Initial state: [2, 0, 1]
        # Expected calculation:
        # result[0] = a[indices[0]] -> a[2] -> 1
        # result[1] = a[indices[1]] -> a[0] -> 2
        # result[2] = a[indices[2]] -> a[1] -> 0
        # Final expected: [1, 2, 0]

        data = np.array([2, 0, 1], dtype=np.intp)
        expected = np.take_along_axis(data, data, axis=0)

        result = npi.take_along_axis(data, data, out=data, axis=0)

        np.testing.assert_array_equal(result, expected)
        assert result is data

    def test_non_contiguous_out(self):
        """Test writing to a non-contiguous 'out' array."""
        a = np.array([[10, 30, 20], [60, 40, 50]])
        indices = np.argsort(a, axis=1)

        full_out = np.zeros((2, 6), dtype=a.dtype)
        out_view = full_out[:, ::2]

        target_out = out_view[:, :3]

        npi.take_along_axis(a, indices, out=target_out, axis=1)

        expected = np.take_along_axis(a, indices, axis=1)
        np.testing.assert_array_equal(target_out, expected)

    def test_out_of_bounds(self):
        """Behavior when indices are out of bounds."""
        a = np.arange(5)
        indices = np.array([1, 6, 2, 3, 4])

        with pytest.raises(IndexError):
            npi.take_along_axis(a, indices, axis=0)

    def test_dimension_mismatch(self):
        """Reject arrays whose shapes do not match along non-axis dimensions."""
        a = np.arange(12).reshape(3, 4)
        indices = np.array([[0, 1], [2, 3]])

        with pytest.raises(ValueError, match="Dimension mismatch"):
            npi.take_along_axis(a, indices, axis=1)


class TestMultidimensionalAndStrides:
    def test_3d_complex_axis(self):
        """Perform take_along_axis on the middle axis (axis=1) of a 3D array."""
        a = np.arange(24).reshape(2, 3, 4)
        indices = np.argsort(a, axis=1)
        expected = np.take_along_axis(a, indices, axis=1)

        result = npi.take_along_axis(a, indices, axis=1)

        np.testing.assert_array_equal(result, expected)
        assert result.shape == (2, 3, 4)

    def test_sliced_non_contiguous_input(self):
        """Take elements from a sliced, non-contiguous input array 'a'."""
        base_a = np.arange(20).reshape(4, 5)
        a = base_a[::2, ::2]

        indices = np.argsort(a, axis=0)
        expected = np.take_along_axis(a, indices, axis=0)

        result = npi.take_along_axis(a, indices, axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_swapaxes_input(self):
        """Input array with a specialized layout via 'swapaxes'."""
        a_orig = np.arange(24).reshape(2, 3, 4)
        a = a_orig.swapaxes(0, 2)

        indices = np.argsort(a, axis=0)
        expected = np.take_along_axis(a, indices, axis=0)

        result = npi.take_along_axis(a, indices, axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_out_with_swapaxes_view(self):
        """Case specifying a 'swapaxes' view as 'out' (triggers write-back)."""
        a = np.array([[10, 30, 20], [60, 40, 50]])
        indices = np.argsort(a, axis=1)
        expected = np.take_along_axis(a, indices, axis=1)

        out_base = np.zeros((3, 2), dtype=a.dtype)
        out_view = out_base.swapaxes(0, 1)

        result = npi.take_along_axis(a, indices, axis=1, out=out_view)

        np.testing.assert_array_equal(out_view, expected)
        assert result is out_view


class TestTakeAlongAxisInJit:
    def test_jit_normal_take_along_axis(self):
        """Basic functional check with 1D arrays inside JIT."""

        @nb.njit
        def jit_func(a, indices):
            return npi.take_along_axis(a, indices, axis=0)

        a = np.array([0, 1, 2])
        indices = np.array([2, 0, 1])
        expected = np.take_along_axis(a, indices, axis=0)

        result = jit_func(a, indices)
        np.testing.assert_array_equal(result, expected)

    def test_jit_take_along_axis_with_axis(self):
        """Multidimensional arrays with axis specified inside JIT."""

        @nb.njit
        def jit_func(a, indices, axis):
            return npi.take_along_axis(a, indices, axis=axis)

        a = np.arange(12).reshape(3, 4)
        indices = np.argsort(a, axis=1)

        expected_ax1 = np.take_along_axis(a, indices, axis=1)
        result_ax1 = jit_func(a, indices, 1)
        np.testing.assert_array_equal(result_ax1, expected_ax1)

        indices_ax0 = np.argsort(a, axis=0)
        expected_ax0 = np.take_along_axis(a, indices_ax0, axis=0)
        result_ax0 = jit_func(a, indices_ax0, 0)
        np.testing.assert_array_equal(result_ax0, expected_ax0)

    def test_jit_axis_none(self):
        """axis=None behavior inside JIT."""

        @nb.njit
        def jit_func(a, indices):
            return npi.take_along_axis(a, indices, axis=None)

        a = np.array([[10, 30, 20], [60, 40, 50]])
        indices = np.array([0, 2, 1, 2, 0, 1])
        expected = np.take_along_axis(a, indices, axis=None)

        result = jit_func(a, indices)
        np.testing.assert_array_equal(result, expected)

    def test_jit_out_is_provided(self):
        """out argument provided inside JIT."""

        @nb.njit
        def jit_func(a, indices, out):
            return npi.take_along_axis(a, indices, axis=0, out=out)

        a = np.array([0, 1, 2])
        indices = np.array([2, 0, 1])
        out = np.zeros_like(indices)

        expected = np.take_along_axis(a, indices, axis=0)
        result = jit_func(a, indices, out)

        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(out, expected)

    def test_jit_parallel_execution(self):
        """Caller uses parallel=True."""

        @nb.njit(parallel=True)
        def jit_func_parallel(a, indices, axis):
            return npi.take_along_axis(a, indices, axis=axis)

        a = np.arange(24).reshape(2, 3, 4)
        indices = np.argsort(a, axis=1)
        expected = np.take_along_axis(a, indices, axis=1)

        result = jit_func_parallel(a, indices, 1)
        np.testing.assert_array_equal(result, expected)

    def test_jit_out_of_bounds(self):
        """Out-of-bounds indices raise inside JIT."""

        @nb.njit
        def jit_func(a, indices):
            return npi.take_along_axis(a, indices, axis=0)

        a = np.arange(5)
        indices = np.array([1, 6, 2, 3, 4])

        with pytest.raises(IndexError):
            jit_func(a, indices)
