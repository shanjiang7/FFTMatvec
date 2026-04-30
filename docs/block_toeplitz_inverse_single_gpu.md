# Single-GPU Block Toeplitz Inverse

This note documents the current single-GPU implementation of the block
lower-triangular Toeplitz inverse prototype. It is intended as a quick code map
for collaborators who want to understand what the implementation does, where
the hot path lives, and how the data layout changes through one Newton
iteration.

## Problem Setting

The current GPU path assumes the block Toeplitz operator is already normalized:

```text
A_0 = I
```

The code computes the first block column of the inverse:

```text
H = A^{-1} mod z^T
```

where `T = num_blocks`, each coefficient block is `r x r`, and
`r = block_dim`.

Blocks are stored coefficient-major, and each block is column-major:

```text
blocks[t * r * r + col * r + row]
```

This storage convention is documented in
`src/BlockToeplitzInverse.hpp`, in the `BlockToeplitzInverse` class comment.

## Main Code Locations

| Purpose | Location |
| --- | --- |
| Public single-GPU API | `src/BlockToeplitzInverse.hpp`, `BlockToeplitzInverse` |
| Single-GPU workspace definition | `src/BlockToeplitzInverse.hpp`, `BlockToeplitzInverseWorkspace` |
| Workspace setup, cuFFT plans, cuBLAS handle, device buffers | `src/BlockToeplitzInverse.cpp`, `BlockToeplitzInverseWorkspace::setup(...)` |
| One Newton doubling iteration | `src/BlockToeplitzInverse.cpp`, `BlockToeplitzInverse::newton_step_gpu(...)` |
| Warm/hot-path solve loop | `src/BlockToeplitzInverse.cpp`, `BlockToeplitzInverse::invert_preloaded_newton_gpu(...)` |
| Copy normalized coefficients to GPU | `src/BlockToeplitzInverse.cpp`, `BlockToeplitzInverse::load_coefficients_gpu(...)` |
| Copy inverse back to host | `src/BlockToeplitzInverse.cpp`, `BlockToeplitzInverse::copy_inverse_from_workspace(...)` |
| Small CUDA layout/update kernels | `src/util_kernels.cu` and `src/util_kernels.hpp` |
| 2D transpose/reindex helper | `src/utils.cpp`, `Utils::transpose_2d(...)` |
| Strided-batched GEMM helper | `src/utils.cpp`, `Utils::sbgemm(...)` |
| Single-GPU correctness and benchmark tests | `test/test_BlockToeplitzInverse.cpp` |

The high-level call chain for a cold one-shot solve is:

```text
invert_newton_gpu(blocks, num_blocks, block_dim)
  -> BlockToeplitzInverseWorkspace::setup_for_problem(...)
  -> invert_newton_gpu(blocks, ..., workspace)
       -> load_coefficients_gpu(...)
       -> invert_preloaded_newton_gpu(...)
       -> copy_inverse_from_workspace(...)
```

For profiling and repeated solves, the intended path is the warm path:

```text
BlockToeplitzInverseWorkspace workspace;
workspace.setup_for_problem(num_blocks, block_dim);

BlockToeplitzInverse::load_coefficients_gpu(..., workspace);

// Optional warmup.
BlockToeplitzInverse::invert_preloaded_newton_gpu(..., workspace);

// Hot path.
BlockToeplitzInverse::invert_preloaded_newton_gpu(..., workspace);
```

This avoids recreating cuBLAS handles, cuFFT plans, and temporary device
buffers inside the measured region.

## Workspace Ownership

`BlockToeplitzInverseWorkspace` owns all persistent single-GPU resources:

```text
cuBLAS handle
cuFFT forward/inverse plans
shared cuFFT work area
real work buffers
frequency-domain buffers
coefficient buffers A and H
```

Important implementation details:

- cuFFT plans are cached for every Newton FFT length needed by the problem.
- cuFFT auto-allocation is disabled with `cufftSetAutoAllocation(plan, 0)`.
- The code queries plan work sizes and allocates one shared max-size cuFFT
  workspace.
- `d_a_coeff` stores normalized input coefficients.
- `d_h_coeff` stores the current inverse coefficients and is reused across all
  Newton iterations.

Memory estimates are printed by setting:

```bash
BTI_PRINT_WORKSPACE=1
```

The report is generated in `src/BlockToeplitzInverse.cpp` by
`build_workspace_memory_report(...)`.

## Newton Doubling Algorithm

The outer loop is in:

```text
BlockToeplitzInverse::invert_preloaded_newton_gpu(...)
```

It starts from:

```text
H_0 = I
```

and doubles the number of valid coefficients each iteration:

```text
m = 1, 2, 4, ...
m_next = min(2 * m, T)
```

For each iteration, `newton_step_gpu(m, m_next, block_dim, fft_len, workspace)`
updates only the newly exposed coefficient range:

```text
H[m : m_next)
```

The lower range:

```text
H[0 : m)
```

is already correct from the previous iteration and is left resident in
`d_h_coeff`.

Conceptually the Newton update is:

```text
U = A * H
H_new = H * (2I - U)
```

Since `H` is already correct modulo `z^m`, the low-order part of `U` is already
`I`. The implementation therefore builds only the high-order correction:

```text
V_high[t] = -U[t],  t in [m, m_next)
V_high[t] = 0,      otherwise
```

and computes:

```text
H_new_high = H * V_high
```

Only `H[m : m_next)` is unpacked and written back.

The helper kernels for this are:

```text
UtilKernels::build_newton_high_correction_real(...)
UtilKernels::unpack_entry_real_range_to_blocks(...)
```

## One Newton Iteration

The hot single-GPU iteration is implemented in:

```text
src/BlockToeplitzInverse.cpp
BlockToeplitzInverse::newton_step_gpu(...)
```

The operation sequence is:

```text
1. Pack coefficient-major A[0:m_next) and H[0:m) into entry-major FFT layout.
2. FFT A and H along the Toeplitz coefficient dimension.
3. Transpose FFT output into frequency-major GEMM layout.
4. Batched GEMM: U(f) = A(f) * H(f).
5. Transpose U back to entry-major layout.
6. IFFT U back to coefficient/time layout.
7. Build V_high = -U_high for t in [m, m_next).
8. FFT V_high.
9. Transpose V_high to frequency-major layout.
10. Batched GEMM: H_high(f) = H(f) * V_high(f).
11. Transpose H_high back to entry-major layout.
12. IFFT H_high.
13. Unpack and write only H[m:m_next).
```

This is the code-level structure:

```text
UtilKernels::pack_blocks_to_entry_real(...)
cufftExecD2Z(...)
Utils::transpose_2d(...)
Utils::sbgemm(...)
Utils::transpose_2d(...)
cufftExecZ2D(...)
UtilKernels::build_newton_high_correction_real(...)
cufftExecD2Z(...)
Utils::transpose_2d(...)
Utils::sbgemm(...)
Utils::transpose_2d(...)
cufftExecZ2D(...)
UtilKernels::unpack_entry_real_range_to_blocks(...)
```

## Data Layouts

There are three important layouts in the single-GPU path.

### Coefficient-Major Storage

This is the resident coefficient layout for `A` and `H`:

```text
d_coeff[t * entries + e]
entries = r * r
e = col * r + row
```

This layout is convenient for storing block Toeplitz coefficients and for
copying host input/output.

### Entry-Major FFT Layout

cuFFT is planned so each matrix entry owns a contiguous time sequence:

```text
d_entry_real[e * fft_len + t]
```

The pack kernel converts:

```text
d_coeff[t * entries + e]
  -> d_entry_real[e * fft_len + t]
```

with zero-padding for `t >= active_length`.

After a real-to-complex FFT, cuFFT output is still entry-major:

```text
d_entry_freq[e * freq_len + f]
freq_len = fft_len / 2 + 1
```

### Frequency-Major GEMM Layout

cuBLAS strided-batched GEMM expects one dense `r x r` matrix per frequency.
Therefore the FFT output is transposed into:

```text
d_freq_major[f * entries + e]
```

This conversion is done with:

```text
Utils::transpose_2d(...)
```

Before IFFT, the GEMM output is transposed back:

```text
d_freq_major[f * entries + e]
  -> d_entry_freq[e * freq_len + f]
```

## Important Helper Functions

### `UtilKernels::pack_blocks_to_entry_real`

Location:

```text
src/util_kernels.cu
src/util_kernels.hpp
```

Converts coefficient-major storage to entry-major FFT input and zero-pads the
unused tail.

### `Utils::transpose_2d`

Location:

```text
src/utils.cpp
src/utils.hpp
```

Uses cuBLAS GEAM to transpose between:

```text
entry-major frequency layout <-> frequency-major GEMM layout
```

This replaces earlier custom reindex kernels in the single-GPU and distributed
paths.

### `Utils::sbgemm`

Location:

```text
src/utils.cpp
src/utils.hpp
```

Runs complex strided-batched GEMM over all frequency points:

```text
C(f) = A(f) * B(f)
```

For double precision, this dispatches to cuBLAS complex double GEMM.

### `UtilKernels::build_newton_high_correction_real`

Location:

```text
src/util_kernels.cu
src/util_kernels.hpp
```

Builds the high-order correction:

```text
V_high[t] = -U[t] / fft_len, t in [m, m_next)
```

The division by `fft_len` accounts for cuFFT's unnormalized inverse transform.

### `UtilKernels::unpack_entry_real_range_to_blocks`

Location:

```text
src/util_kernels.cu
src/util_kernels.hpp
```

Writes only the newly computed coefficient range:

```text
H[m : m_next)
```

This avoids rewriting already-correct lower coefficients.

## Correctness Tests

The main single-GPU tests are in:

```text
test/test_BlockToeplitzInverse.cpp
```

Recommended correctness filters:

```bash
./build/Tests/BlockToeplitzInverseTest \
  --gtest_filter='BlockToeplitzInverseTest.CpuReferenceResidual:BlockToeplitzInverseTest.GpuNewtonMatchesCpuReference:BlockToeplitzInverseTest.ScalarToeplitzGpuNewton'
```

Useful layout/update kernel tests:

```bash
./build/Tests/BlockToeplitzInverseTest \
  --gtest_filter='BlockToeplitzInverseTest.DistributedTransposeRoundTrip:BlockToeplitzInverseTest.BuildNewtonHighCorrectionOnlyWritesUpperRange:BlockToeplitzInverseTest.UnpackEntryRealRangeOnlyUpdatesUpperRange'
```

Although `DistributedTransposeRoundTrip` contains "Distributed" in the name,
it tests the same `Utils::transpose_2d(...)` layout conversion used by the
single-GPU hot path.

## Profiling Entry Points

The profiling tests are:

```text
BlockToeplitzInverseTest.BenchmarkNsysLarge
BlockToeplitzInverseTest.BenchmarkNsysWarmLarge
```

`BenchmarkNsysWarmLarge` is the preferred hot-path benchmark. It:

1. creates the workspace,
2. loads coefficients,
3. runs one warmup solve,
4. starts CUDA profiler capture,
5. runs only `invert_preloaded_newton_gpu(...)`,
6. stops capture.

Example:

```bash
BTI_BENCH_T=4096 \
BTI_BENCH_R=256 \
nsys profile \
  --trace=cuda,cublas,nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --stats=true \
  --force-overwrite=true \
  -o bti_T4096_r256_warm \
  ./build/Tests/BlockToeplitzInverseTest \
  --gtest_filter=BlockToeplitzInverseTest.BenchmarkNsysWarmLarge
```

NVTX ranges are emitted per Newton iteration:

```text
newton_1_to_2_fft_4
newton_2_to_4_fft_8
...
```

These ranges are useful for separating the cost of early small iterations from
the final large iterations.

## Current Scope

This document describes only the single-GPU path:

```text
BlockToeplitzInverseWorkspace
BlockToeplitzInverse::newton_step_gpu(...)
BlockToeplitzInverse::invert_preloaded_newton_gpu(...)
```

The distributed path is separate and uses:

```text
BlockToeplitzInverseDistributedLayout
BlockToeplitzInverseDistributedWorkspace
BlockToeplitzInverse::newton_step_distributed_gpu(...)
```

The two paths share many helper kernels and layout conventions, but the
single-GPU path does not require MPI or NCCL.
