#ifndef __BLOCK_TOEPLITZ_INVERSE_HPP__
#define __BLOCK_TOEPLITZ_INVERSE_HPP__

#include <cuda_runtime.h>
#include <vector>

struct BlockToeplitzMultiplyProfile
{
    double total_ms = 0.0;
    double host_pack_ms = 0.0;
    double cuda_alloc_ms = 0.0;
    double h2d_copy_ms = 0.0;
    double cufft_plan_ms = 0.0;
    double cublas_create_ms = 0.0;
    double forward_fft_ms = 0.0;
    double transpose_to_freq_major_ms = 0.0;
    double sbgemm_ms = 0.0;
    double transpose_to_entry_major_ms = 0.0;
    double inverse_fft_ms = 0.0;
    double d2h_copy_ms = 0.0;
    double scale_truncate_ms = 0.0;
    double cleanup_ms = 0.0;
};

struct BlockToeplitzInverseProfile
{
    double total_ms = 0.0;
    double validate_ms = 0.0;
    double a0_inverse_ms = 0.0;
    double normalize_ms = 0.0;
    double a_prefix_ms = 0.0;
    double v_update_ms = 0.0;
    double undo_normalize_ms = 0.0;
    int iterations = 0;
    int multiply_calls = 0;
    BlockToeplitzMultiplyProfile multiply;
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
     */
    static std::vector<double> invert_cpu_reference(
        const std::vector<double> &blocks, int num_blocks, int block_dim);

    /**
     * @brief Compute the first column of A^{-1} using GPU FFT Newton doubling.
     */
    static std::vector<double> invert_newton_gpu(
        const std::vector<double> &blocks, int num_blocks, int block_dim,
        cudaStream_t stream = 0, BlockToeplitzInverseProfile *profile = nullptr);

    /**
     * @brief Compute a truncated matrix-polynomial product on one GPU.
     *
     * Returns left(t) * right(t) modulo t^out_len. Inputs use the same
     * coefficient-major, column-major block layout as invert_newton_gpu.
     */
    static std::vector<double> multiply_truncated_gpu(
        const std::vector<double> &left, int left_len,
        const std::vector<double> &right, int right_len,
        int out_len, int block_dim, int fft_len, cudaStream_t stream = 0,
        BlockToeplitzMultiplyProfile *profile = nullptr);

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
    static std::vector<double> invert_block_cpu(const double *block, int block_dim);
};

#endif // __BLOCK_TOEPLITZ_INVERSE_HPP__
