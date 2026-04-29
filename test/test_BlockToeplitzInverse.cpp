#include "BlockToeplitzInverse.hpp"
#include "util_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
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

bool env_flag(const char *name)
{
    const char *value = std::getenv(name);
    return value != nullptr && std::string(value) != "0";
}

int env_rank()
{
    const char *names[] = {
        "BTI_DIST_WORLD_RANK",
        "SLURM_PROCID",
        "PMI_RANK",
        "OMPI_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_RANK",
    };
    for (const char *name : names)
    {
        const char *value = std::getenv(name);
        if (value)
            return std::stoi(value);
    }
    return 0;
}

} // namespace

TEST(BlockToeplitzInverseTest, NextPow2)
{
    ASSERT_EQ(BlockToeplitzInverse::next_pow2(1), 1);
    ASSERT_EQ(BlockToeplitzInverse::next_pow2(2), 2);
    ASSERT_EQ(BlockToeplitzInverse::next_pow2(3), 4);
    ASSERT_EQ(BlockToeplitzInverse::next_pow2(17), 32);
}

TEST(BlockToeplitzInverseTest, GoodFftLen)
{
    ASSERT_EQ(BlockToeplitzInverse::good_fft_len(1), 1);
    ASSERT_EQ(BlockToeplitzInverse::good_fft_len(6144), 6144);
    ASSERT_EQ(BlockToeplitzInverse::good_fft_len(10240), 10240);
    ASSERT_GE(BlockToeplitzInverse::good_fft_len(6145), 6145);
    ASSERT_LT(BlockToeplitzInverse::good_fft_len(6145), 8192);
}

TEST(BlockToeplitzInverseTest, DistributedLayoutPartitionsBlockEntries)
{
    const int block_dim = 512;
    size_t covered_entries = 0;

    for (int row_rank = 0; row_rank < 2; ++row_rank)
    {
        for (int col_rank = 0; col_rank < 2; ++col_rank)
        {
            const BlockToeplitzInverseDistributedLayout layout =
                BlockToeplitzInverseDistributedLayout::create(
                    block_dim, 2, 2, row_rank, col_rank);
            EXPECT_EQ(layout.local_rows, 256);
            EXPECT_EQ(layout.local_cols, 256);
            EXPECT_EQ(layout.local_entries(), static_cast<size_t>(256) * 256);
            covered_entries += layout.local_entries();
        }
    }

    EXPECT_EQ(covered_entries, static_cast<size_t>(block_dim) * block_dim);
}

TEST(BlockToeplitzInverseTest, DistributedLayoutHandlesUnevenBlockDim)
{
    const int block_dim = 400;
    size_t covered_entries = 0;

    for (int row_rank = 0; row_rank < 3; ++row_rank)
    {
        for (int col_rank = 0; col_rank < 2; ++col_rank)
        {
            const BlockToeplitzInverseDistributedLayout layout =
                BlockToeplitzInverseDistributedLayout::create(
                    block_dim, 3, 2, row_rank, col_rank);
            EXPECT_GE(layout.local_rows, 133);
            EXPECT_LE(layout.local_rows, 134);
            EXPECT_EQ(layout.local_cols, 200);
            covered_entries += layout.local_entries();
        }
    }

    EXPECT_EQ(covered_entries, static_cast<size_t>(block_dim) * block_dim);
    EXPECT_THROW(BlockToeplitzInverseDistributedLayout::create(
                     block_dim, 3, 2, 3, 0),
                 std::invalid_argument);
}

TEST(BlockToeplitzInverseTest, DistributedLayoutReportsLocalMemory)
{
    const BlockToeplitzInverseDistributedLayout layout =
        BlockToeplitzInverseDistributedLayout::create(400, 2, 2, 1, 0);
    const std::string report = layout.memory_report(8192);

    EXPECT_NE(report.find("BlockToeplitzInverseDistributedLayoutMemory"),
              std::string::npos);
    EXPECT_NE(report.find("local_rows:       [200, 400) size=200"),
              std::string::npos);
    EXPECT_NE(report.find("local_cols:       [0, 200) size=200"),
              std::string::npos);
    EXPECT_NE(report.find("tracked_total_no_cufft_work"), std::string::npos);
}

TEST(BlockToeplitzInverseTest, DistributedReindexRoundTrip)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    const int entries = 6;
    const int freq_len = 5;
    const size_t count = static_cast<size_t>(entries) * freq_len;
    std::vector<ComplexD> entry_freq(count);
    for (int e = 0; e < entries; ++e)
    {
        for (int f = 0; f < freq_len; ++f)
        {
            const double value = 100.0 * e + f;
            entry_freq[static_cast<size_t>(e) * freq_len + f] =
                make_cuDoubleComplex(value, -value);
        }
    }

    ComplexD *d_entry_freq = nullptr;
    ComplexD *d_gemm_freq = nullptr;
    ComplexD *d_round_trip = nullptr;
    gpuErrchk(cudaMalloc(&d_entry_freq, count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc(&d_gemm_freq, count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc(&d_round_trip, count * sizeof(ComplexD)));
    gpuErrchk(cudaMemcpy(d_entry_freq, entry_freq.data(), count * sizeof(ComplexD),
                         cudaMemcpyHostToDevice));

    UtilKernels::entry_freq_to_gemm_layout(
        d_entry_freq, d_gemm_freq, freq_len, entries, 0);
    UtilKernels::gemm_freq_to_entry_layout(
        d_gemm_freq, d_round_trip, freq_len, entries, 0);

    std::vector<ComplexD> gemm_freq(count);
    std::vector<ComplexD> round_trip(count);
    gpuErrchk(cudaMemcpy(gemm_freq.data(), d_gemm_freq, count * sizeof(ComplexD),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(round_trip.data(), d_round_trip, count * sizeof(ComplexD),
                         cudaMemcpyDeviceToHost));

    for (int e = 0; e < entries; ++e)
    {
        for (int f = 0; f < freq_len; ++f)
        {
            const size_t entry_idx = static_cast<size_t>(e) * freq_len + f;
            const size_t gemm_idx = static_cast<size_t>(f) * entries + e;
            EXPECT_DOUBLE_EQ(gemm_freq[gemm_idx].x, entry_freq[entry_idx].x);
            EXPECT_DOUBLE_EQ(gemm_freq[gemm_idx].y, entry_freq[entry_idx].y);
            EXPECT_DOUBLE_EQ(round_trip[entry_idx].x, entry_freq[entry_idx].x);
            EXPECT_DOUBLE_EQ(round_trip[entry_idx].y, entry_freq[entry_idx].y);
        }
    }

    gpuErrchk(cudaFree(d_entry_freq));
    gpuErrchk(cudaFree(d_gemm_freq));
    gpuErrchk(cudaFree(d_round_trip));
}

TEST(BlockToeplitzInverseTest, BuildNewtonVLocalRealOnlyAddsOwnedDiagonal)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    const int out_len = 3;
    const int fft_len = 4;
    const int local_row_start = 1;
    const int local_rows = 2;
    const int local_col_start = 0;
    const int local_cols = 3;
    const int entries = local_rows * local_cols;
    const size_t count = static_cast<size_t>(entries) * fft_len;
    std::vector<double> u_ifft(count);
    for (int e = 0; e < entries; ++e)
    {
        for (int t = 0; t < fft_len; ++t)
            u_ifft[static_cast<size_t>(e) * fft_len + t] =
                10.0 * e + t + 1.0;
    }

    double *d_u_ifft = nullptr;
    double *d_v_real = nullptr;
    gpuErrchk(cudaMalloc(&d_u_ifft, count * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_v_real, count * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_u_ifft, u_ifft.data(), count * sizeof(double),
                         cudaMemcpyHostToDevice));

    UtilKernels::build_newton_v_local_real(
        d_u_ifft, d_v_real, out_len, fft_len,
        local_row_start, local_rows, local_col_start, local_cols, 0);

    std::vector<double> v_real(count);
    gpuErrchk(cudaMemcpy(v_real.data(), d_v_real, count * sizeof(double),
                         cudaMemcpyDeviceToHost));

    for (int e = 0; e < entries; ++e)
    {
        const int local_row = e % local_rows;
        const int local_col = e / local_rows;
        const int global_row = local_row_start + local_row;
        const int global_col = local_col_start + local_col;
        for (int t = 0; t < fft_len; ++t)
        {
            double expected = 0.0;
            if (t < out_len)
            {
                expected = -u_ifft[static_cast<size_t>(e) * fft_len + t] /
                           static_cast<double>(fft_len);
                if (t == 0 && global_row == global_col)
                    expected += 2.0;
            }
            EXPECT_DOUBLE_EQ(v_real[static_cast<size_t>(e) * fft_len + t],
                             expected);
        }
    }

    gpuErrchk(cudaFree(d_u_ifft));
    gpuErrchk(cudaFree(d_v_real));
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
    workspace.setup_for_problem(num_blocks, block_dim);

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

TEST(BlockToeplitzInverseTest, DistributedWorkspaceAllocation)
{
    if (!env_flag("BTI_RUN_DISTRIBUTED_WORKSPACE_TEST"))
        GTEST_SKIP() << "Set BTI_RUN_DISTRIBUTED_WORKSPACE_TEST=1 to allocate "
                        "the distributed workspace skeleton.";
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";

    int device_count = 0;
    gpuErrchk(cudaGetDeviceCount(&device_count));

    const int num_blocks = env_int("BTI_BENCH_T", 8192);
    const int block_dim = env_int("BTI_BENCH_R", 400);
    const int proc_rows = env_int("BTI_DIST_PROC_ROWS", 2);
    const int proc_cols = env_int("BTI_DIST_PROC_COLS", 2);
    const int world_rank = env_rank();
    const int row_rank = env_int("BTI_DIST_ROW_RANK", world_rank % proc_rows);
    const int col_rank = env_int("BTI_DIST_COL_RANK", world_rank / proc_rows);
    const int device = env_int("BTI_DIST_DEVICE", world_rank % device_count);

    gpuErrchk(cudaSetDevice(device));

    const BlockToeplitzInverseDistributedLayout layout =
        BlockToeplitzInverseDistributedLayout::create(
            block_dim, proc_rows, proc_cols, row_rank, col_rank);

    if (env_flag("BTI_DIST_REPORT_ONLY"))
    {
        std::cerr << layout.memory_report(num_blocks);
        return;
    }

    if (env_flag("BTI_PRINT_WORKSPACE"))
        std::cerr << layout.memory_report(num_blocks);

    BlockToeplitzInverseDistributedWorkspace workspace;
    workspace.setup_for_problem(num_blocks, layout);
    gpuErrchk(cudaDeviceSynchronize());
}
