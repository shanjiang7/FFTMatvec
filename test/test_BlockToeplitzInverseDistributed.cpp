#include "BlockToeplitzInverse.hpp"
#include "Comm.hpp"
#include "gtest-mpi-listener.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    ::testing::TestEventListener *listener =
        listeners.Release(listeners.default_result_printer());
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(
        listener, MPI_COMM_WORLD));

    int world_size = 1;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    proc_cols = static_cast<int>(std::sqrt(world_size));
    proc_rows = world_size / proc_cols;
    if (proc_rows > proc_cols)
        std::swap(proc_rows, proc_cols);

    return RUN_ALL_TESTS();
}
