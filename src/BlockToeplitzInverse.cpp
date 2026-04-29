#include "BlockToeplitzInverse.hpp"
#include "util_kernels.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define BLOCK_TOEPLITZ_HAS_NVTX 1
#endif
#endif

#ifndef BLOCK_TOEPLITZ_HAS_NVTX
#define BLOCK_TOEPLITZ_HAS_NVTX 0
#endif

namespace
{
class NvtxRange
{
public:
    explicit NvtxRange(const std::string &name)
    {
#if BLOCK_TOEPLITZ_HAS_NVTX
        nvtxRangePushA(name.c_str());
#else
        (void)name;
#endif
    }

    ~NvtxRange()
    {
#if BLOCK_TOEPLITZ_HAS_NVTX
        nvtxRangePop();
#endif
    }
};

size_t block_entry(int block, int entry_count, int entry)
{
    return static_cast<size_t>(block) * entry_count + entry;
}

size_t cm_entry(int row, int col, int block_dim)
{
    return static_cast<size_t>(col) * block_dim + row;
}

void validate_blocks(const std::vector<double> &blocks, int num_blocks, int block_dim)
{
    if (num_blocks <= 0)
        throw std::invalid_argument("num_blocks must be positive.");
    if (block_dim <= 0)
        throw std::invalid_argument("block_dim must be positive.");

    const size_t expected =
        static_cast<size_t>(num_blocks) * block_dim * block_dim;
    if (blocks.size() != expected)
        throw std::invalid_argument(
            "Block vector has size " + std::to_string(blocks.size()) +
            ", expected " + std::to_string(expected) + ".");
}

void set_identity(double *block, int block_dim, double scale = 1.0)
{
    std::fill(block, block + static_cast<size_t>(block_dim) * block_dim, 0.0);
    for (int i = 0; i < block_dim; ++i)
        block[cm_entry(i, i, block_dim)] = scale;
}

void block_gemm_cpu(const double *A, const double *B, double *C, int block_dim,
                    double alpha = 1.0, double beta = 0.0)
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
            const size_t idx = cm_entry(row, col, block_dim);
            C[idx] = alpha * sum + beta * C[idx];
        }
    }
}

bool is_identity_block(const double *block, int block_dim, double tolerance)
{
    for (int col = 0; col < block_dim; ++col)
    {
        for (int row = 0; row < block_dim; ++row)
        {
            const double expected = (row == col) ? 1.0 : 0.0;
            if (std::abs(block[cm_entry(row, col, block_dim)] - expected) > tolerance)
                return false;
        }
    }
    return true;
}

bool is_power_of_two(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

bool has_good_fft_factors(int n)
{
    if (n <= 0)
        return false;
    const int primes[] = {2, 3, 5, 7};
    for (const int prime : primes)
    {
        while (n % prime == 0)
            n /= prime;
    }
    return n == 1;
}

int good_fft_len_impl(int n)
{
    if (n <= 1)
        return 1;
    int candidate = n;
    while (!has_good_fft_factors(candidate))
        ++candidate;
    return candidate;
}

int partition_size(int global_size, int rank, int parts)
{
    if (global_size <= 0)
        throw std::invalid_argument("global_size must be positive.");
    if (parts <= 0)
        throw std::invalid_argument("parts must be positive.");
    if (rank < 0 || rank >= parts)
        throw std::invalid_argument("rank is outside the process grid.");
    const int base = global_size / parts;
    const int remainder = global_size % parts;
    return base + (rank < remainder ? 1 : 0);
}

int partition_start(int global_size, int rank, int parts)
{
    (void)partition_size(global_size, rank, parts);
    const int base = global_size / parts;
    const int remainder = global_size % parts;
    return rank * base + std::min(rank, remainder);
}

std::vector<int> power_two_fft_lengths(int max_fft_len)
{
    std::vector<int> fft_lengths;
    for (int fft_len = 4; fft_len <= max_fft_len; fft_len <<= 1)
        fft_lengths.push_back(fft_len);
    if (fft_lengths.empty())
        fft_lengths.push_back(4);
    return fft_lengths;
}

std::vector<int> newton_fft_lengths(int num_blocks)
{
    if (num_blocks <= 0)
        throw std::invalid_argument("num_blocks must be positive.");

    std::vector<int> fft_lengths;
    int m = 1;
    while (m < num_blocks)
    {
        const int m_next = std::min(2 * m, num_blocks);
        fft_lengths.push_back(good_fft_len_impl(2 * m_next));
        m = m_next;
    }
    if (fft_lengths.empty())
        fft_lengths.push_back(4);

    std::sort(fft_lengths.begin(), fft_lengths.end());
    fft_lengths.erase(std::unique(fft_lengths.begin(), fft_lengths.end()),
                      fft_lengths.end());
    return fft_lengths;
}

bool print_workspace_report_enabled()
{
    const char *value = std::getenv("BTI_PRINT_WORKSPACE");
    return value != nullptr && std::string(value) != "0";
}

long double bytes_to_gib(long double bytes)
{
    return bytes / (1024.0L * 1024.0L * 1024.0L);
}

std::string bytes_string(long double bytes)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(3) << bytes_to_gib(bytes) << " GiB";
    return os.str();
}

std::string join_ints(const std::vector<int> &values)
{
    std::ostringstream os;
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
            os << ", ";
        os << values[i];
    }
    return os.str();
}

void append_buffer_report(std::ostringstream &os, const std::string &name,
                          size_t count, size_t element_size, int copies,
                          long double &tracked_total)
{
    const long double bytes =
        static_cast<long double>(count) * element_size * copies;
    tracked_total += bytes;
    os << "    " << std::left << std::setw(24) << name
       << " count=" << count
       << ", elem=" << element_size
       << ", copies=" << copies
       << ", total=" << bytes_string(bytes) << "\n";
}

std::string build_workspace_memory_report(
    int max_coeff_blocks, int block_dim, int entries, int max_fft_len,
    int max_freq_len, size_t cufft_work_bytes,
    const std::vector<int> &fft_lengths)
{
    const size_t real_count = static_cast<size_t>(entries) * max_fft_len;
    const size_t freq_count = static_cast<size_t>(entries) * max_freq_len;
    const size_t coeff_count = static_cast<size_t>(max_coeff_blocks) * entries;

    std::ostringstream os;
    os << "\n[BlockToeplitzInverseWorkspaceMemory]\n";
    os << "  max_coeff_blocks: " << max_coeff_blocks << "\n";
    os << "  block_dim:        " << block_dim << "\n";
    os << "  entries:          " << entries << "\n";
    os << "  max_fft_len:      " << max_fft_len << "\n";
    os << "  max_freq_len:     " << max_freq_len << "\n";
    os << "  fft_lengths:      " << join_ints(fft_lengths) << "\n";

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t mem_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (mem_status == cudaSuccess)
    {
        os << "  cuda_free_now:    " << bytes_string(free_bytes) << "\n";
        os << "  cuda_total:       " << bytes_string(total_bytes) << "\n";
    }

    long double tracked_total = 0.0L;
    os << "  buffers:\n";
    append_buffer_report(os, "real work buffers", real_count, sizeof(double),
                         3, tracked_total);
    append_buffer_report(os, "freq entry buffer", freq_count, sizeof(ComplexD),
                         1, tracked_total);
    append_buffer_report(os, "freq major buffers", freq_count, sizeof(ComplexD),
                         3, tracked_total);
    append_buffer_report(os, "coefficient buffers", coeff_count, sizeof(double),
                         2, tracked_total);
    append_buffer_report(os, "shared cufft work", 1, cufft_work_bytes,
                         cufft_work_bytes > 0 ? 1 : 0, tracked_total);
    os << "  tracked_total:    " << bytes_string(tracked_total) << "\n";
    os << std::flush;
    return os.str();
}

} // namespace

BlockToeplitzInverseDistributedLayout
BlockToeplitzInverseDistributedLayout::create(
    int requested_global_block_dim, int requested_proc_rows,
    int requested_proc_cols, int requested_row_rank, int requested_col_rank)
{
    BlockToeplitzInverseDistributedLayout layout;
    layout.global_block_dim = requested_global_block_dim;
    layout.proc_rows = requested_proc_rows;
    layout.proc_cols = requested_proc_cols;
    layout.row_rank = requested_row_rank;
    layout.col_rank = requested_col_rank;
    layout.local_row_start = partition_start(
        requested_global_block_dim, requested_row_rank, requested_proc_rows);
    layout.local_rows = partition_size(
        requested_global_block_dim, requested_row_rank, requested_proc_rows);
    layout.local_col_start = partition_start(
        requested_global_block_dim, requested_col_rank, requested_proc_cols);
    layout.local_cols = partition_size(
        requested_global_block_dim, requested_col_rank, requested_proc_cols);
    return layout;
}

size_t BlockToeplitzInverseDistributedLayout::local_entries() const
{
    return static_cast<size_t>(local_rows) * local_cols;
}

size_t BlockToeplitzInverseDistributedLayout::global_entries() const
{
    return static_cast<size_t>(global_block_dim) * global_block_dim;
}

BlockToeplitzInverseWorkspace::~BlockToeplitzInverseWorkspace()
{
    cleanup();
}

void BlockToeplitzInverseWorkspace::cleanup()
{
    for (PlanEntry &plan : plans)
    {
        if (plan.forward_plan)
            cufftSafeCall(cufftDestroy(plan.forward_plan));
        if (plan.inverse_plan)
            cufftSafeCall(cufftDestroy(plan.inverse_plan));
    }
    plans.clear();

    if (d_cufft_work)
        gpuErrchk(cudaFree(d_cufft_work));
    d_cufft_work = nullptr;
    cufft_work_bytes = 0;

    if (cublas_handle)
    {
        cublasSafeCall(cublasDestroy(cublas_handle));
        cublas_handle = nullptr;
    }

    if (d_left_real)
        gpuErrchk(cudaFree(d_left_real));
    if (d_right_real)
        gpuErrchk(cudaFree(d_right_real));
    if (d_out_real)
        gpuErrchk(cudaFree(d_out_real));
    if (d_out_freq_entry)
        gpuErrchk(cudaFree(d_out_freq_entry));
    if (d_left_freq_major)
        gpuErrchk(cudaFree(d_left_freq_major));
    if (d_right_freq_major)
        gpuErrchk(cudaFree(d_right_freq_major));
    if (d_out_freq_major)
        gpuErrchk(cudaFree(d_out_freq_major));
    if (d_a_coeff)
        gpuErrchk(cudaFree(d_a_coeff));
    if (d_h_coeff)
        gpuErrchk(cudaFree(d_h_coeff));

    d_left_real = nullptr;
    d_right_real = nullptr;
    d_out_real = nullptr;
    d_out_freq_entry = nullptr;
    d_left_freq_major = nullptr;
    d_right_freq_major = nullptr;
    d_out_freq_major = nullptr;
    d_a_coeff = nullptr;
    d_h_coeff = nullptr;

    max_fft_len = 0;
    max_coeff_blocks = 0;
    block_dim = 0;
    entries = 0;
    max_freq_len = 0;
    stream = 0;
}

void BlockToeplitzInverseWorkspace::setup(int requested_max_fft_len, int requested_block_dim,
                                          cudaStream_t requested_stream)
{
    setup(requested_max_fft_len, std::max(1, requested_max_fft_len / 2),
          requested_block_dim, requested_stream);
}

void BlockToeplitzInverseWorkspace::setup(int requested_max_fft_len,
                                          int requested_max_coeff_blocks,
                                          int requested_block_dim,
                                          cudaStream_t requested_stream)
{
    if (requested_max_fft_len <= 0)
        throw std::invalid_argument("max_fft_len must be positive.");
    if (!is_power_of_two(requested_max_fft_len))
        throw std::invalid_argument("max_fft_len must be a power of two.");
    setup(power_two_fft_lengths(requested_max_fft_len), requested_max_coeff_blocks,
          requested_block_dim, requested_stream);
}

void BlockToeplitzInverseWorkspace::setup(const std::vector<int> &requested_fft_lengths,
                                          int requested_max_coeff_blocks,
                                          int requested_block_dim,
                                          cudaStream_t requested_stream)
{
    cleanup();

    if (requested_fft_lengths.empty())
        throw std::invalid_argument("At least one FFT length is required.");
    if (requested_max_coeff_blocks <= 0)
        throw std::invalid_argument("max_coeff_blocks must be positive.");
    if (requested_block_dim <= 0)
        throw std::invalid_argument("block_dim must be positive.");

    std::vector<int> fft_lengths = requested_fft_lengths;
    std::sort(fft_lengths.begin(), fft_lengths.end());
    fft_lengths.erase(std::unique(fft_lengths.begin(), fft_lengths.end()),
                      fft_lengths.end());
    for (const int fft_len : fft_lengths)
    {
        if (fft_len <= 0)
            throw std::invalid_argument("FFT lengths must be positive.");
    }

    max_fft_len = fft_lengths.back();
    max_coeff_blocks = requested_max_coeff_blocks;
    block_dim = requested_block_dim;
    entries = block_dim * block_dim;
    max_freq_len = max_fft_len / 2 + 1;
    stream = requested_stream;

    cublasSafeCall(cublasCreate(&cublas_handle));

    const size_t real_count = static_cast<size_t>(entries) * max_fft_len;
    const size_t freq_count = static_cast<size_t>(entries) * max_freq_len;
    const size_t coeff_count = static_cast<size_t>(max_coeff_blocks) * entries;

    for (const int fft_len : fft_lengths)
    {
        PlanEntry plan;
        plan.fft_len = fft_len;
        plan.freq_len = fft_len / 2 + 1;
        int n[1] = {fft_len};
        size_t forward_work_bytes = 0;
        size_t inverse_work_bytes = 0;

        cufftSafeCall(cufftCreate(&plan.forward_plan));
        cufftSafeCall(cufftSetAutoAllocation(plan.forward_plan, 0));
        cufftSafeCall(cufftMakePlanMany(plan.forward_plan, 1, n, nullptr, 1,
                                        fft_len, nullptr, 1, plan.freq_len,
                                        CUFFT_D2Z, entries, &forward_work_bytes));

        cufftSafeCall(cufftCreate(&plan.inverse_plan));
        cufftSafeCall(cufftSetAutoAllocation(plan.inverse_plan, 0));
        cufftSafeCall(cufftMakePlanMany(plan.inverse_plan, 1, n, nullptr, 1,
                                        plan.freq_len, nullptr, 1, fft_len,
                                        CUFFT_Z2D, entries, &inverse_work_bytes));

        cufft_work_bytes = std::max(cufft_work_bytes, forward_work_bytes);
        cufft_work_bytes = std::max(cufft_work_bytes, inverse_work_bytes);
        cufftSafeCall(cufftSetStream(plan.forward_plan, stream));
        cufftSafeCall(cufftSetStream(plan.inverse_plan, stream));
        plans.push_back(plan);
    }

    if (print_workspace_report_enabled())
    {
        std::cerr << build_workspace_memory_report(
            max_coeff_blocks, block_dim, entries, max_fft_len, max_freq_len,
            cufft_work_bytes, fft_lengths);
    }

    gpuErrchk(cudaMalloc((void **)&d_left_real, real_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_right_real, real_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_out_real, real_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_out_freq_entry, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_left_freq_major, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_right_freq_major, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_out_freq_major, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_a_coeff, coeff_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_h_coeff, coeff_count * sizeof(double)));

    if (cufft_work_bytes > 0)
    {
        gpuErrchk(cudaMalloc(&d_cufft_work, cufft_work_bytes));
        for (const PlanEntry &plan : plans)
        {
            cufftSafeCall(cufftSetWorkArea(plan.forward_plan, d_cufft_work));
            cufftSafeCall(cufftSetWorkArea(plan.inverse_plan, d_cufft_work));
        }
    }
}

void BlockToeplitzInverseWorkspace::setup_for_problem(int num_blocks, int requested_block_dim,
                                                      cudaStream_t requested_stream)
{
    setup(newton_fft_lengths(num_blocks), num_blocks, requested_block_dim,
          requested_stream);
}

std::string BlockToeplitzInverseWorkspace::memory_report() const
{
    std::vector<int> fft_lengths;
    fft_lengths.reserve(plans.size());
    for (const PlanEntry &plan : plans)
        fft_lengths.push_back(plan.fft_len);
    return build_workspace_memory_report(
        max_coeff_blocks, block_dim, entries, max_fft_len, max_freq_len,
        cufft_work_bytes, fft_lengths);
}

const BlockToeplitzInverseWorkspace::PlanEntry &
BlockToeplitzInverseWorkspace::get_plan(int requested_fft_len) const
{
    for (const PlanEntry &plan : plans)
    {
        if (plan.fft_len == requested_fft_len)
            return plan;
    }
    throw std::invalid_argument("No cached FFT plan for requested fft_len.");
}

int BlockToeplitzInverse::next_pow2(int n)
{
    if (n <= 1)
        return 1;
    int result = 1;
    while (result < n)
        result <<= 1;
    return result;
}

int BlockToeplitzInverse::good_fft_len(int n)
{
    return good_fft_len_impl(n);
}

std::vector<double> BlockToeplitzInverse::invert_cpu_reference(
    const std::vector<double> &blocks, int num_blocks, int block_dim)
{
    validate_blocks(blocks, num_blocks, block_dim);
    if (!is_identity_block(blocks.data(), block_dim, 1e-12))
        throw std::invalid_argument(
            "invert_cpu_reference expects normalized input with A_0 = I.");

    const int entries = block_dim * block_dim;
    std::vector<double> inverse(static_cast<size_t>(num_blocks) * entries, 0.0);
    set_identity(inverse.data(), block_dim);

    std::vector<double> sum(entries, 0.0);
    std::vector<double> tmp(entries, 0.0);

    for (int k = 1; k < num_blocks; ++k)
    {
        std::fill(sum.begin(), sum.end(), 0.0);
        for (int i = 1; i <= k; ++i)
        {
            const double *A_i = blocks.data() + block_entry(i, entries, 0);
            const double *H_ki = inverse.data() + block_entry(k - i, entries, 0);
            std::fill(tmp.begin(), tmp.end(), 0.0);
            block_gemm_cpu(A_i, H_ki, tmp.data(), block_dim);
            for (int e = 0; e < entries; ++e)
                sum[e] += tmp[e];
        }

        double *H_k = inverse.data() + block_entry(k, entries, 0);
        for (int e = 0; e < entries; ++e)
            H_k[e] = -sum[e];
    }

    return inverse;
}

void BlockToeplitzInverse::newton_step_gpu(
    int m, int m_next, int block_dim, int fft_len,
    BlockToeplitzInverseWorkspace &workspace)
{
    const int entries = block_dim * block_dim;
    const int freq_len = fft_len / 2 + 1;
    const BlockToeplitzInverseWorkspace::PlanEntry &plans = workspace.get_plan(fft_len);
    cudaStream_t stream = workspace.stream;

    double *d_a_real = workspace.d_left_real;
    double *d_h_real = workspace.d_right_real;
    double *d_work_real = workspace.d_out_real;
    ComplexD *d_work_freq_entry = workspace.d_out_freq_entry;
    ComplexD *d_a_freq_major = workspace.d_left_freq_major;
    ComplexD *d_h_freq_major = workspace.d_right_freq_major;
    ComplexD *d_work_freq_major = workspace.d_out_freq_major;
    cublasHandle_t cublas_handle = workspace.cublas_handle;

    UtilKernels::pack_blocks_to_entry_real(
        workspace.d_a_coeff, d_a_real, m_next, fft_len, entries, stream);
    UtilKernels::pack_blocks_to_entry_real(
        workspace.d_h_coeff, d_h_real, m, fft_len, entries, stream);

    cufftSafeCall(cufftExecD2Z(plans.forward_plan, d_a_real, d_work_freq_entry));
    Utils::transpose_2d(Precision::DOUBLE, d_work_freq_entry, d_a_freq_major,
                        freq_len, entries, cublas_handle, stream);
    cufftSafeCall(cufftExecD2Z(plans.forward_plan, d_h_real, d_work_freq_entry));
    Utils::transpose_2d(Precision::DOUBLE, d_work_freq_entry, d_h_freq_major,
                        freq_len, entries, cublas_handle, stream);

    Utils::sbgemm(Precision::DOUBLE, d_a_freq_major, d_h_freq_major,
                  d_work_freq_major, block_dim, block_dim, block_dim,
                  freq_len, cublas_handle, stream);
    Utils::transpose_2d(Precision::DOUBLE, d_work_freq_major,
                        d_work_freq_entry, entries, freq_len,
                        cublas_handle, stream);
    cufftSafeCall(cufftExecZ2D(plans.inverse_plan, d_work_freq_entry,
                               d_work_real));

    UtilKernels::build_newton_v_real(d_work_real, d_h_real, m_next,
                                     fft_len, block_dim, stream);
    cufftSafeCall(cufftExecD2Z(plans.forward_plan, d_h_real,
                               d_work_freq_entry));
    Utils::transpose_2d(Precision::DOUBLE, d_work_freq_entry, d_a_freq_major,
                        freq_len, entries, cublas_handle, stream);

    Utils::sbgemm(Precision::DOUBLE, d_h_freq_major, d_a_freq_major,
                  d_work_freq_major, block_dim, block_dim, block_dim,
                  freq_len, cublas_handle, stream);
    Utils::transpose_2d(Precision::DOUBLE, d_work_freq_major,
                        d_work_freq_entry, entries, freq_len,
                        cublas_handle, stream);
    cufftSafeCall(cufftExecZ2D(plans.inverse_plan, d_work_freq_entry,
                               d_work_real));
    UtilKernels::unpack_entry_real_to_blocks(
        d_work_real, workspace.d_h_coeff, m_next, fft_len, entries, stream);
}

std::vector<double> BlockToeplitzInverse::invert_newton_gpu(
    const std::vector<double> &blocks, int num_blocks, int block_dim,
    cudaStream_t stream)
{
    validate_blocks(blocks, num_blocks, block_dim);
    if (!is_identity_block(blocks.data(), block_dim, 1e-12))
        throw std::invalid_argument(
            "invert_newton_gpu expects normalized input with A_0 = I.");

    if (num_blocks == 1)
    {
        std::vector<double> H(static_cast<size_t>(block_dim) * block_dim, 0.0);
        set_identity(H.data(), block_dim);
        return H;
    }

    BlockToeplitzInverseWorkspace workspace;
    workspace.setup_for_problem(num_blocks, block_dim, stream);
    return invert_newton_gpu(blocks, num_blocks, block_dim, workspace);
}

std::vector<double> BlockToeplitzInverse::invert_newton_gpu(
    const std::vector<double> &blocks, int num_blocks, int block_dim,
    BlockToeplitzInverseWorkspace &workspace)
{
    load_coefficients_gpu(blocks, num_blocks, block_dim, workspace);
    invert_preloaded_newton_gpu(num_blocks, block_dim, workspace);
    return copy_inverse_from_workspace(num_blocks, block_dim, workspace);
}

void BlockToeplitzInverse::load_coefficients_gpu(
    const std::vector<double> &blocks, int num_blocks, int block_dim,
    BlockToeplitzInverseWorkspace &workspace)
{
    validate_blocks(blocks, num_blocks, block_dim);
    if (!is_identity_block(blocks.data(), block_dim, 1e-12))
        throw std::invalid_argument(
            "invert_newton_gpu expects normalized input with A_0 = I.");
    if (workspace.block_dim != block_dim)
        throw std::invalid_argument("Workspace block_dim does not match requested block_dim.");
    if (workspace.max_coeff_blocks < num_blocks)
        throw std::invalid_argument("Workspace max coefficient capacity is too small.");

    const int max_fft_len = good_fft_len(2 * num_blocks);
    if (workspace.max_fft_len < max_fft_len)
        throw std::invalid_argument("Workspace max_fft_len is too small.");
    for (const int fft_len : newton_fft_lengths(num_blocks))
        (void)workspace.get_plan(fft_len);
    if (!workspace.cublas_handle || !workspace.d_a_coeff || !workspace.d_h_coeff)
        throw std::invalid_argument("Workspace has not been set up.");

    const int entries = block_dim * block_dim;
    cudaStream_t stream = workspace.stream;
    gpuErrchk(cudaMemcpyAsync(workspace.d_a_coeff, blocks.data(),
                              static_cast<size_t>(num_blocks) * entries *
                                  sizeof(double),
                              cudaMemcpyHostToDevice, stream));
}

void BlockToeplitzInverse::invert_preloaded_newton_gpu(
    int num_blocks, int block_dim, BlockToeplitzInverseWorkspace &workspace)
{
    if (num_blocks <= 0)
        throw std::invalid_argument("num_blocks must be positive.");
    if (block_dim <= 0)
        throw std::invalid_argument("block_dim must be positive.");
    if (workspace.block_dim != block_dim)
        throw std::invalid_argument("Workspace block_dim does not match requested block_dim.");
    if (workspace.max_coeff_blocks < num_blocks)
        throw std::invalid_argument("Workspace max coefficient capacity is too small.");

    const int max_fft_len = good_fft_len(2 * num_blocks);
    if (workspace.max_fft_len < max_fft_len)
        throw std::invalid_argument("Workspace max_fft_len is too small.");
    for (const int fft_len : newton_fft_lengths(num_blocks))
        (void)workspace.get_plan(fft_len);
    if (!workspace.cublas_handle || !workspace.d_a_coeff || !workspace.d_h_coeff)
        throw std::invalid_argument("Workspace has not been set up.");

    UtilKernels::set_identity_block(workspace.d_h_coeff, block_dim, workspace.stream);

    int m = 1;
    while (m < num_blocks)
    {
        const int m_next = std::min(2 * m, num_blocks);
        const int fft_len = good_fft_len(2 * m_next);
        const NvtxRange range("newton_" + std::to_string(m) + "_to_" +
                              std::to_string(m_next) + "_fft_" +
                              std::to_string(fft_len));
        newton_step_gpu(m, m_next, block_dim, fft_len, workspace);
        m = m_next;
    }
}

std::vector<double> BlockToeplitzInverse::copy_inverse_from_workspace(
    int num_blocks, int block_dim, BlockToeplitzInverseWorkspace &workspace)
{
    if (num_blocks <= 0)
        throw std::invalid_argument("num_blocks must be positive.");
    if (block_dim <= 0)
        throw std::invalid_argument("block_dim must be positive.");
    if (workspace.block_dim != block_dim)
        throw std::invalid_argument("Workspace block_dim does not match requested block_dim.");
    if (workspace.max_coeff_blocks < num_blocks)
        throw std::invalid_argument("Workspace max coefficient capacity is too small.");
    if (!workspace.d_h_coeff)
        throw std::invalid_argument("Workspace has not been set up.");

    const int entries = block_dim * block_dim;
    std::vector<double> H(static_cast<size_t>(num_blocks) * entries, 0.0);

    gpuErrchk(cudaStreamSynchronize(workspace.stream));
    gpuErrchk(cudaMemcpy(H.data(), workspace.d_h_coeff,
                         static_cast<size_t>(num_blocks) * entries *
                             sizeof(double),
                         cudaMemcpyDeviceToHost));

    return H;
}

double BlockToeplitzInverse::residual_norm(
    const std::vector<double> &blocks, const std::vector<double> &inverse_blocks,
    int num_blocks, int block_dim)
{
    validate_blocks(blocks, num_blocks, block_dim);
    validate_blocks(inverse_blocks, num_blocks, block_dim);

    const int entries = block_dim * block_dim;
    std::vector<double> accum(entries, 0.0);
    std::vector<double> tmp(entries, 0.0);
    double norm_sq = 0.0;

    for (int k = 0; k < num_blocks; ++k)
    {
        std::fill(accum.begin(), accum.end(), 0.0);
        for (int i = 0; i <= k; ++i)
        {
            const double *A_i = blocks.data() + block_entry(i, entries, 0);
            const double *H_ki = inverse_blocks.data() + block_entry(k - i, entries, 0);
            std::fill(tmp.begin(), tmp.end(), 0.0);
            block_gemm_cpu(A_i, H_ki, tmp.data(), block_dim);
            for (int e = 0; e < entries; ++e)
                accum[e] += tmp[e];
        }

        if (k == 0)
        {
            for (int diag = 0; diag < block_dim; ++diag)
                accum[cm_entry(diag, diag, block_dim)] -= 1.0;
        }

        for (double val : accum)
            norm_sq += val * val;
    }

    return std::sqrt(norm_sq);
}
