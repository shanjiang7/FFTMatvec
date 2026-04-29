#include "BlockToeplitzInverse.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__has_include)
#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#define BLOCK_TOEPLITZ_HAS_CUDA_PROFILER_API 1
#endif
#endif

#ifndef BLOCK_TOEPLITZ_HAS_CUDA_PROFILER_API
#define BLOCK_TOEPLITZ_HAS_CUDA_PROFILER_API 0
#endif

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

std::vector<double> make_normalized_problem(int num_blocks, int block_dim,
                                            double coeff_scale = 1.0)
{
    const int entries = block_dim * block_dim;
    std::vector<double> blocks(static_cast<size_t>(num_blocks) * entries, 0.0);

    for (int diag = 0; diag < block_dim; ++diag)
        blocks[cm_entry(diag, diag, block_dim)] = 1.0;

    for (int t = 1; t < num_blocks; ++t)
    {
        for (int col = 0; col < block_dim; ++col)
        {
            for (int row = 0; row < block_dim; ++row)
            {
                const double val =
                    coeff_scale *
                    (0.02 * std::sin(0.7 * (t + 1) * (row + 1)) +
                     0.015 * std::cos(0.4 * (t + 2) * (col + 1)));
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

void profiler_start()
{
#if BLOCK_TOEPLITZ_HAS_CUDA_PROFILER_API
    gpuErrchk(cudaProfilerStart());
#endif
}

void profiler_stop()
{
#if BLOCK_TOEPLITZ_HAS_CUDA_PROFILER_API
    gpuErrchk(cudaProfilerStop());
#endif
}

int env_int(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    if (!value)
        return fallback;
    try
    {
        return std::stoi(value);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument(std::string("Invalid integer environment variable ") +
                                    name + "=" + value);
    }
}

double env_double(const char *name, double fallback)
{
    const char *value = std::getenv(name);
    if (!value)
        return fallback;
    try
    {
        return std::stod(value);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument(std::string("Invalid floating-point environment variable ") +
                                    name + "=" + value);
    }
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
    const std::vector<double> A = make_normalized_problem(num_blocks, block_dim);

    const std::vector<double> H =
        BlockToeplitzInverse::invert_cpu_reference(A, num_blocks, block_dim);

    ASSERT_LT(BlockToeplitzInverse::residual_norm(A, H, num_blocks, block_dim), 1e-11);
}

TEST(BlockToeplitzInverseTest, GpuNewtonRequiresIdentityA0)
{
    const int num_blocks = 4;
    const int block_dim = 2;
    const std::vector<double> A = make_problem(num_blocks, block_dim);

    EXPECT_THROW(BlockToeplitzInverse::invert_cpu_reference(A, num_blocks, block_dim),
                 std::invalid_argument);
    EXPECT_THROW(BlockToeplitzInverse::invert_newton_gpu(A, num_blocks, block_dim),
                 std::invalid_argument);
}

TEST(BlockToeplitzInverseTest, GpuNewtonMatchesCpuReference)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    const int num_blocks = 8;
    const int block_dim = 3;
    const std::vector<double> A = make_normalized_problem(num_blocks, block_dim);

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
    A[0] = 1.0;
    for (int t = 1; t < num_blocks; ++t)
        A[t] = 0.03 / (t + 1);

    const std::vector<double> H_cpu =
        BlockToeplitzInverse::invert_cpu_reference(A, num_blocks, block_dim);
    const std::vector<double> H_gpu =
        BlockToeplitzInverse::invert_newton_gpu(A, num_blocks, block_dim);

    ASSERT_LT(max_abs_diff(H_cpu, H_gpu), 1e-9);
    ASSERT_LT(BlockToeplitzInverse::residual_norm(A, H_gpu, num_blocks, block_dim), 1e-9);
}

TEST(BlockToeplitzInverseTest, BenchmarkSmall)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    const int num_blocks = 128;
    const int block_dim = 16;
    const std::vector<double> A = make_normalized_problem(num_blocks, block_dim, 0.05);

    const std::vector<double> H =
        BlockToeplitzInverse::invert_newton_gpu(A, num_blocks, block_dim);

    const double residual = BlockToeplitzInverse::residual_norm(A, H, num_blocks, block_dim);
    RecordProperty("residual", residual);

    ASSERT_LT(residual, 1e-7);
}

TEST(BlockToeplitzInverseTest, BenchmarkNsysLarge)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    const int num_blocks = env_int("BTI_BENCH_T", 4096);
    const int block_dim = env_int("BTI_BENCH_R", 256);
    const double coeff_scale = env_double("BTI_BENCH_COEFF_SCALE", 0.001);
    const std::vector<double> A =
        make_normalized_problem(num_blocks, block_dim, coeff_scale);

    const std::vector<double> H =
        BlockToeplitzInverse::invert_newton_gpu(A, num_blocks, block_dim);

    ASSERT_EQ(H.size(), static_cast<size_t>(num_blocks) * block_dim * block_dim);
    ASSERT_NEAR(H[0], 1.0, 1e-10);
    ASSERT_TRUE(std::isfinite(H.back()));
}

TEST(BlockToeplitzInverseTest, BenchmarkNsysWarmLarge)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    const int num_blocks = env_int("BTI_BENCH_T", 4096);
    const int block_dim = env_int("BTI_BENCH_R", 256);
    const double coeff_scale = env_double("BTI_BENCH_COEFF_SCALE", 0.001);
    const std::vector<double> A =
        make_normalized_problem(num_blocks, block_dim, coeff_scale);

    BlockToeplitzInverseWorkspace workspace;
    workspace.setup(BlockToeplitzInverse::next_pow2(2 * num_blocks), num_blocks,
                    block_dim);

    BlockToeplitzInverse::load_coefficients_gpu(A, num_blocks, block_dim, workspace);
    BlockToeplitzInverse::invert_preloaded_newton_gpu(num_blocks, block_dim, workspace);
    gpuErrchk(cudaDeviceSynchronize());

    profiler_start();
    BlockToeplitzInverse::invert_preloaded_newton_gpu(num_blocks, block_dim, workspace);
    gpuErrchk(cudaDeviceSynchronize());
    profiler_stop();

    const std::vector<double> H =
        BlockToeplitzInverse::copy_inverse_from_workspace(num_blocks, block_dim, workspace);

    ASSERT_EQ(H.size(), static_cast<size_t>(num_blocks) * block_dim * block_dim);
    ASSERT_NEAR(H[0], 1.0, 1e-10);
    ASSERT_TRUE(std::isfinite(H.back()));
}
