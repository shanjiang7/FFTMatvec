#include "BlockToeplitzInverse.hpp"
#include "util_kernels.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

} // namespace

BlockToeplitzInverseWorkspace::~BlockToeplitzInverseWorkspace()
{
    cleanup();
}

void BlockToeplitzInverseWorkspace::cleanup()
{
    for (PlanEntry &plan : plans)
    {
        cufftSafeCall(cufftDestroy(plan.forward_plan));
        cufftSafeCall(cufftDestroy(plan.inverse_plan));
    }
    plans.clear();

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
    if (d_left_freq_entry)
        gpuErrchk(cudaFree(d_left_freq_entry));
    if (d_right_freq_entry)
        gpuErrchk(cudaFree(d_right_freq_entry));
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
    d_left_freq_entry = nullptr;
    d_right_freq_entry = nullptr;
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
    cleanup();

    if (requested_max_fft_len <= 0)
        throw std::invalid_argument("max_fft_len must be positive.");
    if (!is_power_of_two(requested_max_fft_len))
        throw std::invalid_argument("max_fft_len must be a power of two.");
    if (requested_block_dim <= 0)
        throw std::invalid_argument("block_dim must be positive.");

    max_fft_len = requested_max_fft_len;
    max_coeff_blocks = std::max(1, max_fft_len / 2);
    block_dim = requested_block_dim;
    entries = block_dim * block_dim;
    max_freq_len = max_fft_len / 2 + 1;
    stream = requested_stream;

    cublasSafeCall(cublasCreate(&cublas_handle));

    const size_t real_count = static_cast<size_t>(entries) * max_fft_len;
    const size_t freq_count = static_cast<size_t>(entries) * max_freq_len;
    const size_t coeff_count = static_cast<size_t>(max_coeff_blocks) * entries;

    gpuErrchk(cudaMalloc((void **)&d_left_real, real_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_right_real, real_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_out_real, real_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_left_freq_entry, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_right_freq_entry, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_out_freq_entry, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_left_freq_major, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_right_freq_major, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_out_freq_major, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_a_coeff, coeff_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_h_coeff, coeff_count * sizeof(double)));

    for (int fft_len = 4; fft_len <= max_fft_len; fft_len <<= 1)
    {
        PlanEntry plan;
        plan.fft_len = fft_len;
        plan.freq_len = fft_len / 2 + 1;
        int n[1] = {fft_len};
        cufftSafeCall(cufftPlanMany(&plan.forward_plan, 1, n, nullptr, 1, fft_len,
                                    nullptr, 1, plan.freq_len, CUFFT_D2Z, entries));
        cufftSafeCall(cufftPlanMany(&plan.inverse_plan, 1, n, nullptr, 1, plan.freq_len,
                                    nullptr, 1, fft_len, CUFFT_Z2D, entries));
        cufftSafeCall(cufftSetStream(plan.forward_plan, stream));
        cufftSafeCall(cufftSetStream(plan.inverse_plan, stream));
        plans.push_back(plan);
    }
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

std::vector<double> BlockToeplitzInverse::invert_block_cpu(
    const double *block, int block_dim)
{
    const int width = 2 * block_dim;
    std::vector<double> aug(static_cast<size_t>(block_dim) * width, 0.0);

    for (int row = 0; row < block_dim; ++row)
    {
        for (int col = 0; col < block_dim; ++col)
            aug[static_cast<size_t>(row) * width + col] =
                block[cm_entry(row, col, block_dim)];
        aug[static_cast<size_t>(row) * width + block_dim + row] = 1.0;
    }

    for (int pivot_col = 0; pivot_col < block_dim; ++pivot_col)
    {
        int pivot_row = pivot_col;
        double pivot_abs = std::abs(aug[static_cast<size_t>(pivot_row) * width + pivot_col]);
        for (int row = pivot_col + 1; row < block_dim; ++row)
        {
            const double candidate =
                std::abs(aug[static_cast<size_t>(row) * width + pivot_col]);
            if (candidate > pivot_abs)
            {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if (pivot_abs < 1e-14)
            throw std::runtime_error("A_0 is singular or numerically singular.");

        if (pivot_row != pivot_col)
        {
            for (int col = 0; col < width; ++col)
            {
                std::swap(aug[static_cast<size_t>(pivot_col) * width + col],
                          aug[static_cast<size_t>(pivot_row) * width + col]);
            }
        }

        const double pivot = aug[static_cast<size_t>(pivot_col) * width + pivot_col];
        for (int col = 0; col < width; ++col)
            aug[static_cast<size_t>(pivot_col) * width + col] /= pivot;

        for (int row = 0; row < block_dim; ++row)
        {
            if (row == pivot_col)
                continue;
            const double factor = aug[static_cast<size_t>(row) * width + pivot_col];
            if (factor == 0.0)
                continue;
            for (int col = 0; col < width; ++col)
            {
                aug[static_cast<size_t>(row) * width + col] -=
                    factor * aug[static_cast<size_t>(pivot_col) * width + col];
            }
        }
    }

    std::vector<double> inverse(static_cast<size_t>(block_dim) * block_dim, 0.0);
    for (int row = 0; row < block_dim; ++row)
    {
        for (int col = 0; col < block_dim; ++col)
        {
            inverse[cm_entry(row, col, block_dim)] =
                aug[static_cast<size_t>(row) * width + block_dim + col];
        }
    }
    return inverse;
}

std::vector<double> BlockToeplitzInverse::invert_cpu_reference(
    const std::vector<double> &blocks, int num_blocks, int block_dim)
{
    validate_blocks(blocks, num_blocks, block_dim);

    const int entries = block_dim * block_dim;
    const std::vector<double> A0_inv = invert_block_cpu(blocks.data(), block_dim);
    std::vector<double> inverse(static_cast<size_t>(num_blocks) * entries, 0.0);
    std::copy(A0_inv.begin(), A0_inv.end(), inverse.begin());

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
        std::fill(tmp.begin(), tmp.end(), 0.0);
        block_gemm_cpu(A0_inv.data(), sum.data(), tmp.data(), block_dim);
        for (int e = 0; e < entries; ++e)
            H_k[e] = -tmp[e];
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
    ComplexD *d_a_freq_entry = workspace.d_left_freq_entry;
    ComplexD *d_h_freq_entry = workspace.d_right_freq_entry;
    ComplexD *d_work_freq_entry = workspace.d_out_freq_entry;
    ComplexD *d_a_freq_major = workspace.d_left_freq_major;
    ComplexD *d_h_freq_major = workspace.d_right_freq_major;
    ComplexD *d_work_freq_major = workspace.d_out_freq_major;
    cublasHandle_t cublas_handle = workspace.cublas_handle;

    UtilKernels::pack_blocks_to_entry_real(
        workspace.d_a_coeff, d_a_real, m_next, fft_len, entries, stream);
    UtilKernels::pack_blocks_to_entry_real(
        workspace.d_h_coeff, d_h_real, m, fft_len, entries, stream);

    cufftSafeCall(cufftExecD2Z(plans.forward_plan, d_a_real, d_a_freq_entry));
    cufftSafeCall(cufftExecD2Z(plans.forward_plan, d_h_real, d_h_freq_entry));

    Utils::transpose_2d(Precision::DOUBLE, d_a_freq_entry, d_a_freq_major,
                        freq_len, entries, cublas_handle, stream);
    Utils::transpose_2d(Precision::DOUBLE, d_h_freq_entry, d_h_freq_major,
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
    workspace.setup(next_pow2(2 * num_blocks), block_dim, stream);
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

    const int max_fft_len = next_pow2(2 * num_blocks);
    if (workspace.max_fft_len < max_fft_len)
        throw std::invalid_argument("Workspace max_fft_len is too small.");
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

    const int max_fft_len = next_pow2(2 * num_blocks);
    if (workspace.max_fft_len < max_fft_len)
        throw std::invalid_argument("Workspace max_fft_len is too small.");
    if (!workspace.cublas_handle || !workspace.d_a_coeff || !workspace.d_h_coeff)
        throw std::invalid_argument("Workspace has not been set up.");

    UtilKernels::set_identity_block(workspace.d_h_coeff, block_dim, workspace.stream);

    int m = 1;
    while (m < num_blocks)
    {
        const int m_next = std::min(2 * m, num_blocks);
        const int fft_len = next_pow2(2 * m_next);
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
