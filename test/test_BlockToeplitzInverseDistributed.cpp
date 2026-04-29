#include "BlockToeplitzInverse.hpp"
#include "Comm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int proc_rows = 1;
int proc_cols = 1;

namespace
{
size_t cm_entry(int row, int col, int rows)
{
    return static_cast<size_t>(col) * rows + row;
}

ComplexD add_complex(ComplexD a, ComplexD b)
{
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

ComplexD mul_complex(ComplexD a, ComplexD b)
{
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y,
                                a.x * b.y + a.y * b.x);
}

ComplexD make_a_value(int freq, int row, int col)
{
    return make_cuDoubleComplex(
        0.25 * (freq + 1) + 0.5 * row - 0.125 * col,
        0.01 * (freq + row + 2 * col + 1));
}

ComplexD make_b_value(int freq, int row, int col)
{
    return make_cuDoubleComplex(
        0.125 * (freq + 2) - 0.25 * row + 0.375 * col,
        -0.02 * (2 * freq + row - col + 1));
}

bool cuda_available()
{
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool env_flag(const char *name)
{
    const char *value = std::getenv(name);
    return value != nullptr && std::string(value) != "0";
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

long double bytes_to_gib(long double bytes)
{
    return bytes / (1024.0L * 1024.0L * 1024.0L);
}

std::vector<ComplexD> make_local_tile(
    const std::vector<ComplexD> &global_freq_major, int block_dim, int freq_len,
    const BlockToeplitzInverseDistributedLayout &layout)
{
    const size_t local_entries = layout.local_entries();
    std::vector<ComplexD> local(static_cast<size_t>(freq_len) * local_entries);
    for (int f = 0; f < freq_len; ++f)
    {
        for (int local_col = 0; local_col < layout.local_cols; ++local_col)
        {
            for (int local_row = 0; local_row < layout.local_rows; ++local_row)
            {
                const int global_row = layout.local_row_start + local_row;
                const int global_col = layout.local_col_start + local_col;
                local[static_cast<size_t>(f) * local_entries +
                      cm_entry(local_row, local_col, layout.local_rows)] =
                    global_freq_major[static_cast<size_t>(f) * block_dim * block_dim +
                                      cm_entry(global_row, global_col, block_dim)];
            }
        }
    }
    return local;
}

std::vector<ComplexD> reference_local_product(
    const std::vector<ComplexD> &global_a,
    const std::vector<ComplexD> &global_b,
    int block_dim, int freq_len,
    const BlockToeplitzInverseDistributedLayout &layout)
{
    const size_t local_entries = layout.local_entries();
    std::vector<ComplexD> local(static_cast<size_t>(freq_len) * local_entries,
                                make_cuDoubleComplex(0.0, 0.0));
    for (int f = 0; f < freq_len; ++f)
    {
        for (int local_col = 0; local_col < layout.local_cols; ++local_col)
        {
            for (int local_row = 0; local_row < layout.local_rows; ++local_row)
            {
                const int global_row = layout.local_row_start + local_row;
                const int global_col = layout.local_col_start + local_col;
                ComplexD sum = make_cuDoubleComplex(0.0, 0.0);
                for (int inner = 0; inner < block_dim; ++inner)
                {
                    const ComplexD a =
                        global_a[static_cast<size_t>(f) * block_dim * block_dim +
                                 cm_entry(global_row, inner, block_dim)];
                    const ComplexD b =
                        global_b[static_cast<size_t>(f) * block_dim * block_dim +
                                 cm_entry(inner, global_col, block_dim)];
                    sum = add_complex(sum, mul_complex(a, b));
                }
                local[static_cast<size_t>(f) * local_entries +
                      cm_entry(local_row, local_col, layout.local_rows)] = sum;
            }
        }
    }
    return local;
}

std::vector<double> make_normalized_problem(int num_blocks, int block_dim)
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
                blocks[static_cast<size_t>(t) * entries +
                       cm_entry(row, col, block_dim)] =
                    0.003 * std::sin(0.5 * (t + 1) * (row + 1)) +
                    0.002 * std::cos(0.25 * (t + 2) * (col + 1));
            }
        }
    }

    return blocks;
}

std::vector<double> make_local_real_tile(
    const std::vector<double> &global_blocks, int num_blocks, int block_dim,
    const BlockToeplitzInverseDistributedLayout &layout)
{
    const size_t local_entries = layout.local_entries();
    std::vector<double> local(static_cast<size_t>(num_blocks) * local_entries);
    for (int t = 0; t < num_blocks; ++t)
    {
        for (int local_col = 0; local_col < layout.local_cols; ++local_col)
        {
            for (int local_row = 0; local_row < layout.local_rows; ++local_row)
            {
                const int global_row = layout.local_row_start + local_row;
                const int global_col = layout.local_col_start + local_col;
                local[static_cast<size_t>(t) * local_entries +
                      cm_entry(local_row, local_col, layout.local_rows)] =
                    global_blocks[static_cast<size_t>(t) * block_dim * block_dim +
                                  cm_entry(global_row, global_col, block_dim)];
            }
        }
    }
    return local;
}

std::vector<double> make_sparse_local_normalized_problem(
    int num_blocks, const BlockToeplitzInverseDistributedLayout &layout,
    double coeff_scale)
{
    const size_t local_entries = layout.local_entries();
    std::vector<double> local(static_cast<size_t>(num_blocks) * local_entries,
                              0.0);

    for (int local_col = 0; local_col < layout.local_cols; ++local_col)
    {
        for (int local_row = 0; local_row < layout.local_rows; ++local_row)
        {
            const int global_row = layout.local_row_start + local_row;
            const int global_col = layout.local_col_start + local_col;
            const size_t entry = cm_entry(local_row, local_col, layout.local_rows);
            if (global_row == global_col)
                local[entry] = 1.0;

            if (num_blocks > 1 && coeff_scale != 0.0)
            {
                local[local_entries + entry] =
                    coeff_scale *
                    (0.25 + 0.001 * (global_row + 1) -
                     0.0005 * (global_col + 1));
            }
        }
    }

    return local;
}

std::vector<double> gather_global_real_tiles(
    const std::vector<double> &local_blocks, int num_blocks, int block_dim,
    int proc_rows, int proc_cols, MPI_Comm comm)
{
    int world_rank = 0;
    int world_size = 1;
    MPICHECK(MPI_Comm_rank(comm, &world_rank));
    MPICHECK(MPI_Comm_size(comm, &world_size));

    const int local_count = static_cast<int>(local_blocks.size());
    std::vector<int> counts(world_size, 0);
    MPICHECK(MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1,
                        MPI_INT, 0, comm));

    std::vector<int> displs(world_size, 0);
    int total_count = 0;
    if (world_rank == 0)
    {
        for (int rank = 0; rank < world_size; ++rank)
        {
            displs[rank] = total_count;
            total_count += counts[rank];
        }
    }

    std::vector<double> gathered(total_count);
    MPICHECK(MPI_Gatherv(local_blocks.data(), local_count, MPI_DOUBLE,
                         gathered.data(), counts.data(), displs.data(),
                         MPI_DOUBLE, 0, comm));

    std::vector<double> global;
    if (world_rank != 0)
        return global;

    const int entries = block_dim * block_dim;
    global.assign(static_cast<size_t>(num_blocks) * entries, 0.0);
    for (int rank = 0; rank < world_size; ++rank)
    {
        const int row_rank = rank % proc_rows;
        const int col_rank = rank / proc_rows;
        const BlockToeplitzInverseDistributedLayout rank_layout =
            BlockToeplitzInverseDistributedLayout::create(
                block_dim, proc_rows, proc_cols, row_rank, col_rank);
        const double *rank_data = gathered.data() + displs[rank];
        const size_t rank_entries = rank_layout.local_entries();
        for (int t = 0; t < num_blocks; ++t)
        {
            for (int local_col = 0; local_col < rank_layout.local_cols; ++local_col)
            {
                for (int local_row = 0; local_row < rank_layout.local_rows; ++local_row)
                {
                    const int global_row = rank_layout.local_row_start + local_row;
                    const int global_col = rank_layout.local_col_start + local_col;
                    global[static_cast<size_t>(t) * entries +
                           cm_entry(global_row, global_col, block_dim)] =
                        rank_data[static_cast<size_t>(t) * rank_entries +
                                  cm_entry(local_row, local_col,
                                           rank_layout.local_rows)];
                }
            }
        }
    }

    return global;
}
} // namespace

TEST(BlockToeplitzInverseDistributedTest, SbgemmFreqMajorMatchesReference)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";
    if (proc_rows != proc_cols)
        GTEST_SKIP() << "Distributed GEMM test requires a square process grid.";

    Comm comm(MPI_COMM_WORLD, proc_rows, proc_cols);
    const int block_dim = 4;
    const int freq_len = 3;
    const BlockToeplitzInverseDistributedLayout layout =
        BlockToeplitzInverseDistributedLayout::create(
            block_dim, proc_rows, proc_cols,
            comm.get_row_color(), comm.get_col_color());

    std::vector<ComplexD> global_a(
        static_cast<size_t>(freq_len) * block_dim * block_dim);
    std::vector<ComplexD> global_b(global_a.size());
    for (int f = 0; f < freq_len; ++f)
    {
        for (int col = 0; col < block_dim; ++col)
        {
            for (int row = 0; row < block_dim; ++row)
            {
                global_a[static_cast<size_t>(f) * block_dim * block_dim +
                         cm_entry(row, col, block_dim)] =
                    make_a_value(f, row, col);
                global_b[static_cast<size_t>(f) * block_dim * block_dim +
                         cm_entry(row, col, block_dim)] =
                    make_b_value(f, row, col);
            }
        }
    }

    const std::vector<ComplexD> local_a =
        make_local_tile(global_a, block_dim, freq_len, layout);
    const std::vector<ComplexD> local_b =
        make_local_tile(global_b, block_dim, freq_len, layout);
    const std::vector<ComplexD> expected =
        reference_local_product(global_a, global_b, block_dim, freq_len, layout);

    const size_t local_count =
        static_cast<size_t>(freq_len) * layout.local_entries();
    ComplexD *d_a = nullptr;
    ComplexD *d_b = nullptr;
    ComplexD *d_c = nullptr;
    gpuErrchk(cudaMalloc(&d_a, local_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc(&d_b, local_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc(&d_c, local_count * sizeof(ComplexD)));
    gpuErrchk(cudaMemcpyAsync(d_a, local_a.data(),
                              local_count * sizeof(ComplexD),
                              cudaMemcpyHostToDevice, comm.get_stream()));
    gpuErrchk(cudaMemcpyAsync(d_b, local_b.data(),
                              local_count * sizeof(ComplexD),
                              cudaMemcpyHostToDevice, comm.get_stream()));

    BlockToeplitzInverseDistributedWorkspace workspace;
    workspace.setup_for_problem(2, layout, comm.get_stream());
    workspace.sbgemm_freq_major(d_a, d_b, d_c, freq_len, comm);

    std::vector<ComplexD> actual(local_count);
    gpuErrchk(cudaMemcpyAsync(actual.data(), d_c,
                              local_count * sizeof(ComplexD),
                              cudaMemcpyDeviceToHost, comm.get_stream()));
    gpuErrchk(cudaStreamSynchronize(comm.get_stream()));

    for (size_t i = 0; i < local_count; ++i)
    {
        EXPECT_NEAR(actual[i].x, expected[i].x, 1e-10);
        EXPECT_NEAR(actual[i].y, expected[i].y, 1e-10);
    }

    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_c));
}

TEST(BlockToeplitzInverseDistributedTest, NewtonMatchesCpuReference)
{
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";
    if (proc_rows != proc_cols)
        GTEST_SKIP() << "Distributed Newton test requires a square process grid.";

    Comm comm(MPI_COMM_WORLD, proc_rows, proc_cols);
    const int num_blocks = 8;
    const int block_dim = 4;
    const BlockToeplitzInverseDistributedLayout layout =
        BlockToeplitzInverseDistributedLayout::create(
            block_dim, proc_rows, proc_cols,
            comm.get_row_color(), comm.get_col_color());

    const std::vector<double> global_a =
        make_normalized_problem(num_blocks, block_dim);
    const std::vector<double> local_a =
        make_local_real_tile(global_a, num_blocks, block_dim, layout);

    BlockToeplitzInverseDistributedWorkspace workspace;
    workspace.setup_for_problem(num_blocks, layout, comm.get_stream());
    const std::vector<double> local_h =
        BlockToeplitzInverse::invert_newton_distributed_gpu(
            local_a, num_blocks, layout, workspace, comm);

    const std::vector<double> global_h =
        gather_global_real_tiles(local_h, num_blocks, block_dim,
                                 proc_rows, proc_cols, MPI_COMM_WORLD);

    if (comm.get_world_rank() == 0)
    {
        const std::vector<double> expected =
            BlockToeplitzInverse::invert_cpu_reference(
                global_a, num_blocks, block_dim);
        ASSERT_EQ(global_h.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(global_h[i], expected[i], 1e-8);
    }
}

TEST(BlockToeplitzInverseDistributedTest, DistributedLargeSmoke)
{
    if (!env_flag("BTI_RUN_DISTRIBUTED_LARGE_TEST"))
        GTEST_SKIP() << "Set BTI_RUN_DISTRIBUTED_LARGE_TEST=1 to run the "
                        "large distributed Newton smoke test.";
    if (!cuda_available())
        GTEST_SKIP() << "CUDA device is not available.";
    if (proc_rows != proc_cols)
        GTEST_SKIP() << "Large distributed smoke test requires a square process grid.";

    Comm comm(MPI_COMM_WORLD, proc_rows, proc_cols);
    const int num_blocks = env_int("BTI_BENCH_T", 8192);
    const int block_dim = env_int("BTI_BENCH_R", 400);
    const double coeff_scale = env_double("BTI_BENCH_COEFF_SCALE", 0.0);
    const BlockToeplitzInverseDistributedLayout layout =
        BlockToeplitzInverseDistributedLayout::create(
            block_dim, proc_rows, proc_cols,
            comm.get_row_color(), comm.get_col_color());

    if (comm.get_world_rank() == 0)
    {
        const long double local_coeff_bytes =
            static_cast<long double>(num_blocks) * layout.local_entries() *
            sizeof(double);
        std::cerr << "\n[BlockToeplitzInverseDistributedLargeSmoke]\n"
                  << "  num_blocks:       " << num_blocks << "\n"
                  << "  global_block_dim: " << block_dim << "\n"
                  << "  process_grid:     " << proc_rows << " x "
                  << proc_cols << "\n"
                  << "  rank0_local_entries: " << layout.local_entries() << "\n"
                  << "  rank0_local_coeff_host: "
                  << static_cast<double>(bytes_to_gib(local_coeff_bytes))
                  << " GiB\n";
    }

    BlockToeplitzInverseDistributedWorkspace workspace;
    workspace.setup_for_problem(num_blocks, layout, comm.get_stream());
    if (env_flag("BTI_DIST_REPORT_ONLY"))
        return;

    std::vector<double> local_a =
        make_sparse_local_normalized_problem(num_blocks, layout, coeff_scale);
    BlockToeplitzInverse::load_coefficients_distributed_gpu(
        local_a, num_blocks, layout, workspace);
    gpuErrchk(cudaStreamSynchronize(comm.get_stream()));
    std::vector<double>().swap(local_a);

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    BlockToeplitzInverse::invert_preloaded_newton_distributed_gpu(
        num_blocks, layout, workspace, comm);
    gpuErrchk(cudaStreamSynchronize(comm.get_stream()));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double end = MPI_Wtime();

    if (comm.get_world_rank() == 0)
        std::cerr << "  distributed_newton_time_s: " << (end - start) << "\n";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    if (world_rank != 0)
    {
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
        delete listeners.Release(listeners.default_xml_generator());
    }

    proc_cols = static_cast<int>(std::sqrt(world_size));
    proc_rows = world_size / proc_cols;
    if (proc_rows > proc_cols)
        std::swap(proc_rows, proc_cols);

    const int local_result = RUN_ALL_TESTS();
    int global_result = 0;
    MPICHECK(MPI_Allreduce(&local_result, &global_result, 1, MPI_INT,
                           MPI_MAX, MPI_COMM_WORLD));
    MPICHECK(MPI_Finalize());
    return global_result;
}
