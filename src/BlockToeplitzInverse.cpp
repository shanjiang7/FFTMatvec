#include "BlockToeplitzInverse.hpp"
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

} // namespace

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

std::vector<double> BlockToeplitzInverse::multiply_truncated_gpu(
    const std::vector<double> &left, int left_len,
    const std::vector<double> &right, int right_len,
    int out_len, int block_dim, int fft_len, cudaStream_t stream)
{
    if (left_len <= 0 || right_len <= 0 || out_len <= 0)
        throw std::invalid_argument("Polynomial lengths must be positive.");
    if (out_len > fft_len)
        throw std::invalid_argument("out_len must be no larger than fft_len.");
    if (fft_len < left_len + right_len - 1)
        throw std::invalid_argument("fft_len is too small for the requested product.");

    const int entries = block_dim * block_dim;
    if (left.size() < static_cast<size_t>(left_len) * entries ||
        right.size() < static_cast<size_t>(right_len) * entries)
    {
        throw std::invalid_argument("Input polynomial storage is smaller than its length.");
    }

    const int freq_len = fft_len / 2 + 1;
    const size_t real_count = static_cast<size_t>(entries) * fft_len;
    const size_t freq_count = static_cast<size_t>(entries) * freq_len;
    const size_t freq_major_count = static_cast<size_t>(freq_len) * entries;

    std::vector<double> h_left_real(real_count, 0.0);
    std::vector<double> h_right_real(real_count, 0.0);

    for (int t = 0; t < left_len; ++t)
    {
        for (int e = 0; e < entries; ++e)
            h_left_real[static_cast<size_t>(e) * fft_len + t] =
                left[block_entry(t, entries, e)];
    }
    for (int t = 0; t < right_len; ++t)
    {
        for (int e = 0; e < entries; ++e)
            h_right_real[static_cast<size_t>(e) * fft_len + t] =
                right[block_entry(t, entries, e)];
    }

    double *d_left_real = nullptr;
    double *d_right_real = nullptr;
    double *d_out_real = nullptr;
    ComplexD *d_left_freq_entry = nullptr;
    ComplexD *d_right_freq_entry = nullptr;
    ComplexD *d_out_freq_entry = nullptr;
    ComplexD *d_left_freq_major = nullptr;
    ComplexD *d_right_freq_major = nullptr;
    ComplexD *d_out_freq_major = nullptr;
    cufftHandle forward_plan;
    cufftHandle inverse_plan;
    cublasHandle_t cublas_handle = nullptr;

    gpuErrchk(cudaMalloc((void **)&d_left_real, real_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_right_real, real_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_out_real, real_count * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_left_freq_entry, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_right_freq_entry, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_out_freq_entry, freq_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_left_freq_major, freq_major_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_right_freq_major, freq_major_count * sizeof(ComplexD)));
    gpuErrchk(cudaMalloc((void **)&d_out_freq_major, freq_major_count * sizeof(ComplexD)));

    gpuErrchk(cudaMemcpy(d_left_real, h_left_real.data(), real_count * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_right_real, h_right_real.data(), real_count * sizeof(double),
                         cudaMemcpyHostToDevice));

    int n[1] = {fft_len};
    cufftSafeCall(cufftPlanMany(&forward_plan, 1, n, nullptr, 1, fft_len, nullptr, 1,
                                freq_len, CUFFT_D2Z, entries));
    cufftSafeCall(cufftPlanMany(&inverse_plan, 1, n, nullptr, 1, freq_len, nullptr, 1,
                                fft_len, CUFFT_Z2D, entries));
    cufftSafeCall(cufftSetStream(forward_plan, stream));
    cufftSafeCall(cufftSetStream(inverse_plan, stream));

    cufftSafeCall(cufftExecD2Z(forward_plan, d_left_real, d_left_freq_entry));
    cufftSafeCall(cufftExecD2Z(forward_plan, d_right_real, d_right_freq_entry));

    cublasSafeCall(cublasCreate(&cublas_handle));
    Utils::transpose_2d(Precision::DOUBLE, d_left_freq_entry, d_left_freq_major,
                        freq_len, entries, cublas_handle, stream);
    Utils::transpose_2d(Precision::DOUBLE, d_right_freq_entry, d_right_freq_major,
                        freq_len, entries, cublas_handle, stream);

    Utils::sbgemm(Precision::DOUBLE, d_left_freq_major, d_right_freq_major, d_out_freq_major,
                  block_dim, block_dim, block_dim, freq_len, cublas_handle, stream);
    Utils::transpose_2d(Precision::DOUBLE, d_out_freq_major, d_out_freq_entry,
                        entries, freq_len, cublas_handle, stream);

    cufftSafeCall(cufftExecZ2D(inverse_plan, d_out_freq_entry, d_out_real));
    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<double> h_out_real(real_count);
    gpuErrchk(cudaMemcpy(h_out_real.data(), d_out_real, real_count * sizeof(double),
                         cudaMemcpyDeviceToHost));

    std::vector<double> out(static_cast<size_t>(out_len) * entries, 0.0);
    const double scale = 1.0 / static_cast<double>(fft_len);
    for (int t = 0; t < out_len; ++t)
    {
        for (int e = 0; e < entries; ++e)
        {
            out[block_entry(t, entries, e)] =
                h_out_real[static_cast<size_t>(e) * fft_len + t] * scale;
        }
    }

    cublasSafeCall(cublasDestroy(cublas_handle));
    cufftSafeCall(cufftDestroy(forward_plan));
    cufftSafeCall(cufftDestroy(inverse_plan));
    gpuErrchk(cudaFree(d_left_real));
    gpuErrchk(cudaFree(d_right_real));
    gpuErrchk(cudaFree(d_out_real));
    gpuErrchk(cudaFree(d_left_freq_entry));
    gpuErrchk(cudaFree(d_right_freq_entry));
    gpuErrchk(cudaFree(d_out_freq_entry));
    gpuErrchk(cudaFree(d_left_freq_major));
    gpuErrchk(cudaFree(d_right_freq_major));
    gpuErrchk(cudaFree(d_out_freq_major));

    return out;
}

std::vector<double> BlockToeplitzInverse::invert_newton_gpu(
    const std::vector<double> &blocks, int num_blocks, int block_dim,
    cudaStream_t stream)
{
    validate_blocks(blocks, num_blocks, block_dim);

    const int entries = block_dim * block_dim;
    const std::vector<double> A0_inv = invert_block_cpu(blocks.data(), block_dim);

    std::vector<double> normalized(static_cast<size_t>(num_blocks) * entries, 0.0);
    set_identity(normalized.data(), block_dim);
    for (int k = 1; k < num_blocks; ++k)
    {
        block_gemm_cpu(A0_inv.data(), blocks.data() + block_entry(k, entries, 0),
                       normalized.data() + block_entry(k, entries, 0), block_dim);
    }

    std::vector<double> H(entries, 0.0);
    set_identity(H.data(), block_dim);

    int m = 1;
    while (m < num_blocks)
    {
        const int m_next = std::min(2 * m, num_blocks);
        const int fft_len = next_pow2(2 * m_next);

        std::vector<double> A_prefix(normalized.begin(),
                                     normalized.begin() + static_cast<size_t>(m_next) * entries);
        std::vector<double> U = multiply_truncated_gpu(
            A_prefix, m_next, H, m, m_next, block_dim, fft_len, stream);

        std::vector<double> V(U.size(), 0.0);
        for (size_t i = 0; i < U.size(); ++i)
            V[i] = -U[i];
        for (int diag = 0; diag < block_dim; ++diag)
            V[cm_entry(diag, diag, block_dim)] += 2.0;

        H = multiply_truncated_gpu(H, m, V, m_next, m_next, block_dim, fft_len, stream);
        m = m_next;
    }

    std::vector<double> inverse(static_cast<size_t>(num_blocks) * entries, 0.0);
    for (int k = 0; k < num_blocks; ++k)
    {
        block_gemm_cpu(H.data() + block_entry(k, entries, 0), A0_inv.data(),
                       inverse.data() + block_entry(k, entries, 0), block_dim);
    }

    return inverse;
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
