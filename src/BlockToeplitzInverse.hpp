#ifndef __BLOCK_TOEPLITZ_INVERSE_HPP__
#define __BLOCK_TOEPLITZ_INVERSE_HPP__

#include "shared.hpp"
#include <vector>

class BlockToeplitzInverseWorkspace
{
private:
    struct PlanEntry
    {
        int fft_len = 0;
        int freq_len = 0;
        cufftHandle forward_plan;
        cufftHandle inverse_plan;
    };

    int max_fft_len = 0;
    int max_coeff_blocks = 0;
    int block_dim = 0;
    int entries = 0;
    int max_freq_len = 0;
    cudaStream_t stream = 0;
    cublasHandle_t cublas_handle = nullptr;
    std::vector<PlanEntry> plans;

    double *d_left_real = nullptr;
    double *d_right_real = nullptr;
    double *d_out_real = nullptr;
    ComplexD *d_left_freq_entry = nullptr;
    ComplexD *d_right_freq_entry = nullptr;
    ComplexD *d_out_freq_entry = nullptr;
    ComplexD *d_left_freq_major = nullptr;
    ComplexD *d_right_freq_major = nullptr;
    ComplexD *d_out_freq_major = nullptr;
    double *d_a_coeff = nullptr;
    double *d_h_coeff = nullptr;

    const PlanEntry &get_plan(int fft_len) const;

public:
    BlockToeplitzInverseWorkspace() = default;
    ~BlockToeplitzInverseWorkspace();

    BlockToeplitzInverseWorkspace(const BlockToeplitzInverseWorkspace &) = delete;
    BlockToeplitzInverseWorkspace &operator=(const BlockToeplitzInverseWorkspace &) = delete;

    void setup(int max_fft_len, int block_dim, cudaStream_t stream = 0);
    void cleanup();

    friend class BlockToeplitzInverse;
};

/**
 * @brief Utilities for inverting block lower-triangular Toeplitz matrices.
 *
 * Blocks are stored coefficient-major. Each block is an r x r matrix in
 * cuBLAS-compatible column-major layout:
 *
 *     blocks[k * r * r + col * r + row]
 *
 * The current GPU path is a single-GPU, in-memory Newton doubling prototype.
 */
class BlockToeplitzInverse
{
public:
    /**
     * @brief Compute the first column of A^{-1} using a direct CPU recurrence.
     *
     * This reference path follows the same normalized problem setting as the
     * GPU path and requires A_0 = I.
     */
    static std::vector<double> invert_cpu_reference(
        const std::vector<double> &blocks, int num_blocks, int block_dim);

    /**
     * @brief Compute the first column of A^{-1} using GPU FFT Newton doubling.
     *
     * This path assumes the input has already been normalized so A_0 = I.
     */
    static std::vector<double> invert_newton_gpu(
        const std::vector<double> &blocks, int num_blocks, int block_dim,
        cudaStream_t stream = 0);

    /**
     * @brief Compute A^{-1} using a caller-owned workspace.
     *
     * The workspace must have been set up with max_fft_len >= next_pow2(2*num_blocks)
     * and the same block_dim. This path is intended for warm benchmarks and repeated
     * solves where cuBLAS handles, cuFFT plans, and GPU buffers should be reused.
     */
    static std::vector<double> invert_newton_gpu(
        const std::vector<double> &blocks, int num_blocks, int block_dim,
        BlockToeplitzInverseWorkspace &workspace);

    /**
     * @brief Copy normalized A coefficients into a caller-owned GPU workspace.
     */
    static void load_coefficients_gpu(
        const std::vector<double> &blocks, int num_blocks, int block_dim,
        BlockToeplitzInverseWorkspace &workspace);

    /**
     * @brief Run Newton doubling using A already loaded in the workspace.
     *
     * The result remains resident in the workspace. Use copy_inverse_from_workspace
     * when a host vector is needed.
     */
    static void invert_preloaded_newton_gpu(
        int num_blocks, int block_dim, BlockToeplitzInverseWorkspace &workspace);

    /**
     * @brief Copy the device-resident inverse from the workspace to a host vector.
     */
    static std::vector<double> copy_inverse_from_workspace(
        int num_blocks, int block_dim, BlockToeplitzInverseWorkspace &workspace);

    /**
     * @brief Frobenius norm of A * H - I modulo t^num_blocks.
     */
    static double residual_norm(
        const std::vector<double> &blocks, const std::vector<double> &inverse_blocks,
        int num_blocks, int block_dim);

    /**
     * @brief Small helper exposed for tests and iteration planning.
     */
    static int next_pow2(int n);

private:
    static void newton_step_gpu(
        int m, int m_next, int block_dim, int fft_len,
        BlockToeplitzInverseWorkspace &workspace);
};

#endif // __BLOCK_TOEPLITZ_INVERSE_HPP__
