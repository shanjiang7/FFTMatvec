#include "BlockToeplitzInverse.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace
{
size_t block_entry(int block, int entry_count, int entry)
{
    return static_cast<size_t>(block) * entry_count + entry;
}

size_t cm_entry(int row, int col, int block_dim)
{
    return static_cast<size_t>(col) * block_dim + row;
}

std::vector<double> make_problem(int num_blocks, int block_dim)
{
    const int entries = block_dim * block_dim;
    std::vector<double> blocks(static_cast<size_t>(num_blocks) * entries, 0.0);

    for (int col = 0; col < block_dim; ++col)
    {
        for (int row = 0; row < block_dim; ++row)
        {
            const double diag = (row == col) ? (2.0 + 0.25 * row) : 0.0;
            const double off_diag = (row == col) ? 0.0 : 0.05 * (row + 1) - 0.03 * (col + 1);
            blocks[cm_entry(row, col, block_dim)] = diag + off_diag;
        }
    }

    for (int t = 1; t < num_blocks; ++t)
    {
        for (int col = 0; col < block_dim; ++col)
        {
            for (int row = 0; row < block_dim; ++row)
            {
                const double val = 0.02 * std::sin(0.7 * (t + 1) * (row + 1)) +
                                   0.015 * std::cos(0.4 * (t + 2) * (col + 1));
                blocks[block_entry(t, entries, cm_entry(row, col, block_dim))] = val;
            }
        }
    }

    return blocks;
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    return max_diff;
}

bool cuda_available()
{
    int device_count = 0;
    const cudaError_t device_err = cudaGetDeviceCount(&device_count);
    return device_err == cudaSuccess && device_count > 0;
}

std::vector<double> make_polynomial(int num_blocks, int block_dim, double offset)
{
    const int entries = block_dim * block_dim;
    std::vector<double> blocks(static_cast<size_t>(num_blocks) * entries, 0.0);

    for (int t = 0; t < num_blocks; ++t)
    {
        for (int col = 0; col < block_dim; ++col)
        {
            for (int row = 0; row < block_dim; ++row)
            {
                const double val = 0.04 * std::sin(offset + 0.9 * (t + 1) * (row + 1)) +
                                   0.03 * std::cos(offset + 0.5 * (t + 2) * (col + 1));
                blocks[block_entry(t, entries, cm_entry(row, col, block_dim))] = val;
            }
        }
    }

    return blocks;
}

void block_gemm_cpu(const double *A, const double *B, double *C, int block_dim)
{
    for (int col = 0; col < block_dim; ++col)
    {
        for (int row = 0; row < block_dim; ++row)
        {
            double sum = 0.0;
            for (int inner = 0; inner < block_dim; ++inner)
            {
                sum += A[cm_entry(row, inner, block_dim)] *
                       B[cm_entry(inner, col, block_dim)];
            }
            C[cm_entry(row, col, block_dim)] = sum;
        }
    }
}

std::vector<double> multiply_truncated_cpu(const std::vector<double> &left, int left_len,
                                           const std::vector<double> &right, int right_len,
                                           int out_len, int block_dim)
{
    const int entries = block_dim * block_dim;
    std::vector<double> out(static_cast<size_t>(out_len) * entries, 0.0);
    std::vector<double> tmp(entries, 0.0);

    for (int k = 0; k < out_len; ++k)
    {
        for (int i = 0; i <= k; ++i)
        {
            const int j = k - i;
            if (i >= left_len || j >= right_len)
                continue;

            block_gemm_cpu(left.data() + block_entry(i, entries, 0),
                           right.data() + block_entry(j, entries, 0),
                           tmp.data(), block_dim);
            for (int e = 0; e < entries; ++e)
                out[block_entry(k, entries, e)] += tmp[e];
        }
    }

    return out;
}

void expect_polynomial_multiply_matches_cpu(int left_len, int right_len, int out_len,
                                            int block_dim, double tolerance)
{
    const std::vector<double> left = make_polynomial(left_len, block_dim, 0.2);
    const std::vector<double> right = make_polynomial(right_len, block_dim, 1.1);
    const int fft_len = BlockToeplitzInverse::next_pow2(left_len + right_len - 1);

    const std::vector<double> expected =
        multiply_truncated_cpu(left, left_len, right, right_len, out_len, block_dim);
    const std::vector<double> actual =
        BlockToeplitzInverse::multiply_truncated_gpu(
            left, left_len, right, right_len, out_len, block_dim, fft_len);

    ASSERT_LT(max_abs_diff(expected, actual), tolerance);
}
} // namespace

TEST(BlockToeplitzInverseTest, NextPow2)
{
    ASSERT_EQ(BlockToeplitzInverse::next_pow2(1), 1);
    ASSERT_EQ(BlockToeplitzInverse::next_pow2(2), 2);
    ASSERT_EQ(BlockToeplitzInverse::next_pow2(3), 4);
    ASSERT_EQ(BlockToeplitzInverse::next_pow2(17), 32);
}

TEST(BlockToeplitzInverseTest, CpuReferenceResidual)
{
    const int num_blocks = 8;
    const int block_dim = 3;
    const std::vector<double> A = make_problem(num_blocks, block_dim);

    const std::vector<double> H =
        BlockToeplitzInverse::invert_cpu_reference(A, num_blocks, block_dim);

    ASSERT_LT(BlockToeplitzInverse::residual_norm(A, H, num_blocks, block_dim), 1e-11);
}

TEST(BlockToeplitzInverseTest, PolynomialMultiplyScalarTruncated)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    expect_polynomial_multiply_matches_cpu(5, 4, 4, 1, 1e-10);
}

TEST(BlockToeplitzInverseTest, PolynomialMultiplyUnequalLengths)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    expect_polynomial_multiply_matches_cpu(3, 5, 7, 2, 1e-10);
}

TEST(BlockToeplitzInverseTest, PolynomialMultiplyMatrixBlocks)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    expect_polynomial_multiply_matches_cpu(4, 3, 5, 4, 1e-10);
}

TEST(BlockToeplitzInverseTest, GpuNewtonMatchesCpuReference)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    const int num_blocks = 8;
    const int block_dim = 3;
    const std::vector<double> A = make_problem(num_blocks, block_dim);

    const std::vector<double> H_cpu =
        BlockToeplitzInverse::invert_cpu_reference(A, num_blocks, block_dim);
    const std::vector<double> H_gpu =
        BlockToeplitzInverse::invert_newton_gpu(A, num_blocks, block_dim);

    ASSERT_LT(max_abs_diff(H_cpu, H_gpu), 1e-8);
    ASSERT_LT(BlockToeplitzInverse::residual_norm(A, H_gpu, num_blocks, block_dim), 1e-8);
}

TEST(BlockToeplitzInverseTest, ScalarToeplitzGpuNewton)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    const int num_blocks = 16;
    const int block_dim = 1;
    std::vector<double> A(static_cast<size_t>(num_blocks), 0.0);
    A[0] = 1.7;
    for (int t = 1; t < num_blocks; ++t)
        A[t] = 0.03 / (t + 1);

    const std::vector<double> H_cpu =
        BlockToeplitzInverse::invert_cpu_reference(A, num_blocks, block_dim);
    const std::vector<double> H_gpu =
        BlockToeplitzInverse::invert_newton_gpu(A, num_blocks, block_dim);

    ASSERT_LT(max_abs_diff(H_cpu, H_gpu), 1e-9);
    ASSERT_LT(BlockToeplitzInverse::residual_norm(A, H_gpu, num_blocks, block_dim), 1e-9);
}
