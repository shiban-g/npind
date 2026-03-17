import numba as nb
import numpy as np
import pytest

import npind as npi


class TestTakeEdgeCases:
    def test_normal_take(self):
        """Basic functional check with 1D arrays."""
        a = np.array([10, 20, 30, 40, 50])
        indices = np.array([0, 1, 4])
        expected = np.take(a, indices, axis=0)
        np.testing.assert_array_equal(npi.take(a, indices, axis=0), expected)

    def test_out_is_a(self):
        """Case where 'a' and 'out' are the same array (In-place behavior)."""
        # Test case where the array size remains unchanged
        a = np.array([10, 20, 30, 40, 50])
        indices = np.array([4, 3, 2, 1, 0])
        expected = np.take(a, indices, axis=0)

        # Pass 'a' as the 'out' argument
        result = npi.take(a, indices, out=a, axis=0)

        # Verify if 'a' itself is modified and the result matches the expectation
        np.testing.assert_array_equal(a, expected)
        np.testing.assert_array_equal(result, expected)
        assert result is a

    def test_out_is_indices(self):
        """Case where 'indices' and 'out' are the same array."""
        # Note: 'indices' must be of integer type to be valid indices
        a = np.array([10, 20, 30, 40, 50])
        indices = np.array([0, 2, 4], dtype=np.intp)
        expected = np.take(a, indices, axis=0)

        # Pass 'indices' as 'out'.
        # Since 'a' is int, the dtypes match and no casting is required here.
        result = npi.take(a, indices, out=indices, axis=0)

        np.testing.assert_array_equal(result, expected)
        assert result is indices

    def test_all_same_array(self):
        """[Extreme Edge Case] When 'a', 'indices', and 'out' are all the same object."""
        # For this to be valid:
        # 1. Shapes must be identical.
        # 2. Dtypes must be compatible (integer types).
        # 3. Chaotic scenario: using own values as its own indices.

        # Initial state: [2, 0, 1]
        # Expected calculation:
        # result[0] = a[indices[0]] -> a[2] -> 1
        # result[1] = a[indices[1]] -> a[0] -> 2
        # result[2] = a[indices[2]] -> a[1] -> 0
        # Final expected: [1, 2, 0]

        data = np.array([2, 0, 1], dtype=np.intp)
        expected = np.take(data, data, axis=0)

        # Pass the same object for all three parameters
        result = npi.take(data, data, out=data, axis=0)

        np.testing.assert_array_equal(result, expected)
        assert result is data

    def test_non_contiguous_out(self):
        """Test writing to a non-contiguous 'out' array."""
        a = np.arange(10)
        indices = np.array([1, 3, 5])

        # Create an 'out' buffer with non-contiguous memory layout
        full_out = np.zeros(10, dtype=a.dtype)
        out_view = full_out[::2]  # A view with stride of 2 (size 5)

        # Slice to the required size of 3
        target_out = out_view[:3]

        npi.take(a, indices, out=target_out, axis=0)

        # Verify if values are correctly written into the original full buffer
        expected_full = np.take(a, indices, axis=0)
        np.testing.assert_array_equal(target_out, expected_full)

    @pytest.mark.parametrize("mode", ["raise"])
    def test_out_of_bounds(self, mode):
        """Behavior when indices are out of bounds."""
        a = np.arange(5)
        indices = np.array([1, 6])  # 6 is out of bounds

        with pytest.raises(nb.errors.NumbaIndexError):
            npi.take(a, indices, mode=mode, axis=0)


class TestMultidimensionalAndStrides:
    def test_2d_axis_1(self):
        """Perform 'take' on a 2D array along axis=1."""
        # (3, 4) array
        a = np.arange(12).reshape(3, 4)
        indices = np.array([3, 0, 1])
        # Ground truth from numpy.take
        expected = np.take(a, indices, axis=1)

        result = npi.take(a, indices, axis=1)

        np.testing.assert_array_equal(result, expected)

    def test_3d_complex_axis(self):
        """Perform 'take' on the middle axis (axis=1) of a 3D array."""
        # (2, 3, 4) array
        a = np.arange(24).reshape(2, 3, 4)
        indices = np.array([2, 0])
        expected = np.take(a, indices, axis=1)

        result = npi.take(a, indices, axis=1)

        np.testing.assert_array_equal(result, expected)
        # Verify result shape is (2, 2, 4)
        assert result.shape == (2, 2, 4)

    def test_sliced_non_contiguous_input(self):
        """Take elements from a sliced, non-contiguous input array 'a'."""
        # Create non-contiguous array using a stride-2 slice
        base_a = np.arange(20).reshape(4, 5)
        a = base_a[::2, ::2]  # Shape (2, 3), fragmented in memory

        indices = np.array([1, 0])
        expected = np.take(a, indices, axis=0)

        result = npi.take(a, indices, axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_swapaxes_input(self):
        """Input array with a specialized layout via 'swapaxes'."""
        a_orig = np.arange(24).reshape(2, 3, 4)
        # Swap the 0th and 2nd axes
        a = a_orig.swapaxes(0, 2)  # Shape (4, 3, 2)

        indices = np.array([1, 3])
        expected = np.take(a, indices, axis=0)

        result = npi.take(a, indices, axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_multidim_indices(self):
        """Behavior when 'indices' itself is multidimensional."""
        # numpy.take reflects the shape of 'indices' in the output shape
        a = np.array([10, 20, 30, 40])
        indices = np.array([[0, 1], [2, 3]])  # 2x2

        # Expected: [[10, 20], [30, 40]]
        expected = np.take(a, indices, axis=0)

        result = npi.take(a, indices, axis=0)

        np.testing.assert_array_equal(result, expected)
        assert result.shape == (2, 2)

    def test_out_with_swapaxes_view(self):
        """Case specifying a 'swapaxes' view as 'out' (triggers write-back)."""
        a = np.arange(6).reshape(2, 3)
        indices = np.array([2, 0, 1])  # axis=1

        # Expected result shape is (2, 3)
        expected = np.take(a, indices, axis=1)

        # Create 'out' as (3, 2), then swap axes to present it as (2, 3)
        out_base = np.zeros((3, 2), dtype=a.dtype)
        out_view = out_base.swapaxes(0, 1)  # (2, 3) view

        result = npi.take(a, indices, axis=1, out=out_view)

        # Check if values are correctly placed in the original buffer through the view
        np.testing.assert_array_equal(out_view, expected)
        # Verify if the original reference is maintained (or written back correctly)
        assert result is out_view


class TestTakeInJit:
    def test_jit_normal_take(self):
        """1次元配列に対する基本的な機能テスト（JIT内）"""

        @nb.njit
        def jit_func(a, indices):
            return npi.take(a, indices)

        a = np.array([10, 20, 30, 40, 50])
        indices = np.array([0, 1, 4])
        expected = np.take(a, indices)

        result = jit_func(a, indices)
        np.testing.assert_array_equal(result, expected)

    def test_jit_take_with_axis(self):
        """axisを指定した場合の多次元配列のテスト（JIT内）"""

        @nb.njit
        def jit_func(a, indices, axis):
            return npi.take(a, indices, axis=axis)

        a = np.arange(12).reshape(3, 4)
        indices = np.array([2, 0, 1])

        # axis = 1
        expected_ax1 = np.take(a, indices, axis=1)
        result_ax1 = jit_func(a, indices, 1)
        np.testing.assert_array_equal(result_ax1, expected_ax1)

        # axis = 0
        expected_ax0 = np.take(a, indices, axis=0)
        result_ax0 = jit_func(a, indices, 0)
        np.testing.assert_array_equal(result_ax0, expected_ax0)

    def test_jit_out_is_provided(self):
        """out引数を提供した場合のテスト（JIT内）"""

        @nb.njit
        def jit_func(a, indices, out):
            return npi.take(a, indices, out=out)

        a = np.array([10, 20, 30, 40, 50])
        indices = np.array([0, 1, 4])
        # Numba内で out を使い回すため、事前に確保して渡す
        out = np.zeros_like(indices)

        expected = np.take(a, indices)
        result = jit_func(a, indices, out)

        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(out, expected)
        # Numba の関数越しでも元のバッファが書き換えられているか確認

    def test_jit_multidim_indices(self):
        """indicesが多次元配列の場合のテスト（JIT内）"""

        @nb.njit
        def jit_func(a, indices, axis):
            return npi.take(a, indices, axis=axis)

        a = np.arange(24).reshape(2, 3, 4)
        indices = np.array([[0, 1], [1, 0]])

        expected = np.take(a, indices, axis=2)
        result = jit_func(a, indices, 2)

        np.testing.assert_array_equal(result, expected)
        assert result.shape == expected.shape

    def test_jit_parallel_execution(self):
        """呼び出し側が parallel=True の場合のテスト"""

        # ユーザーが並列ループ内で使った場合や、JIT環境で並列化をオンにした場合を想定
        @nb.njit(parallel=True)
        def jit_func_parallel(a, indices, axis):
            return npi.take(a, indices, axis=axis)

        a = np.arange(24).reshape(2, 3, 4)
        indices = np.array([2, 0, 1])
        expected = np.take(a, indices, axis=1)

        result = jit_func_parallel(a, indices, 1)
        np.testing.assert_array_equal(result, expected)

    def test_jit_out_of_bounds(self):
        """インデックスが範囲外の場合のエラーテスト（JIT内）"""

        @nb.njit
        def jit_func(a, indices):
            return npi.take(a, indices)

        a = np.arange(5)
        indices = np.array([1, 6])  # 6 は範囲外

        # Numba の実行時エラーは Python 側でラップされるため、
        # 汎用的な Exception でキャッチするか、より具体的なエラー型を指定します
        with pytest.raises(Exception):
            jit_func(a, indices)
