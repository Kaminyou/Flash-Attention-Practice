#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void flash_attn_v1_kernel(
    const float *Q, const float *K, const float *V,
    const int N, const int d, const int Tc, const int Tr,
    const int Bc, const int Br, const float softmax_scale,
    float *l, float *m, float *O)
{
    const int bx = blockIdx.x;  // batch index
    const int by = blockIdx.y;  // head index
    const int tx = threadIdx.x;  // br index
    const int nh = gridDim.y;  // # of head

    // Calculate base offsets
    const int qkv_offset = (bx * nh * N * d) + (by * N * d);
    const int lm_offset  = (bx * nh * N) + (by * N);

    // Allocate shared memory (SRAM)
    extern __shared__ float sram[];
    const int KV_TILE_SIZE = Bc * d;
    const int Q_TILE_SIZE  = Br * d;

    float *Qi = sram;
    float *Kj = Qi + Q_TILE_SIZE;
    float *Vj = Kj + KV_TILE_SIZE;
    float *S  = Vj + KV_TILE_SIZE;

    // Iterate over Tc
    for (int j = 0; j < Tc; j++) {
        // Load Kj and Vj into shared memory
        for (int x = 0; x < d; ++x) {
            int idx = tx * d + x;
            Kj[idx] = K[qkv_offset + j * KV_TILE_SIZE + idx];
            Vj[idx] = V[qkv_offset + j * KV_TILE_SIZE + idx];
        }
        __syncthreads();

        // Iterate over Tr
        for (int i = 0; i < Tr; i++) {
            if (tx < Br) {
                // Load Qi into shared memory
                for (int x = 0; x < d; ++x) {
                    int idx = tx * d + x;
                    Qi[idx] = Q[qkv_offset + i * Q_TILE_SIZE + idx];
                }

                // Load previous row l and m
                int row_idx = lm_offset + i * Br + tx;
                float row_m_prev = m[row_idx];
                float row_l_prev = l[row_idx];

                // Compute S = Qi * Kj^T (scaled dot-product)
                float row_m = -INFINITY;
                for (int y = 0; y < Bc; ++y) {
                    float dot = 0.0f;
                    for (int x = 0; x < d; ++x)
                        dot += Qi[tx * d + x] * Kj[y * d + x];

                    float scaled = dot * softmax_scale;
                    S[tx * Bc + y] = scaled;
                    row_m = fmaxf(row_m, scaled);
                }

                // Compute softmax: P = exp(S - row_m)
                float row_l = 0.0f;
                for (int y = 0; y < Bc; ++y) {
                    float exp_val = __expf(S[tx * Bc + y] - row_m);
                    S[tx * Bc + y] = exp_val;
                    row_l += exp_val;
                }

                // Compute updated m and l using numerically stable update
                float row_m_new = fmaxf(row_m_prev, row_m);
                float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                                  __expf(row_m - row_m_new) * row_l;

                // Compute and update output O
                for (int x = 0; x < d; ++x) {
                    float weighted_sum = 0.0f;
                    for (int y = 0; y < Bc; ++y)
                        weighted_sum += S[tx * Bc + y] * Vj[y * d + x];

                    int out_idx = qkv_offset + i * Q_TILE_SIZE + tx * d + x;
                    float prev_O = O[out_idx];
                    O[out_idx] = (1.0f / row_l_new) * (
                        row_l_prev * __expf(row_m_prev - row_m_new) * prev_O +
                        __expf(row_m - row_m_new) * weighted_sum);
                }

                // Update l and m
                l[row_idx] = row_l_new;
                m[row_idx] = row_m_new;
            }
        }
        __syncthreads();
    }
}

// PyTorch interface function
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int Bc_arg, int Br_arg) {
    // 1. Input Validation and Type Checking
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");

    // Expected shape: (B, nh, N_seq, D)
    TORCH_CHECK(Q.dim() == 4, "Q must be a 4D tensor (B, nh, N_seq, D)");
    TORCH_CHECK(K.dim() == 4, "K must be a 4D tensor (B, nh, N_seq, D)");
    TORCH_CHECK(V.dim() == 4, "V must be a 4D tensor (B, nh, N_seq, D)");

    int B = Q.size(0);
    int nh = Q.size(1);
    int N_seq = Q.size(2);
    int D = Q.size(3);

    // Ensure dimensions match
    TORCH_CHECK(K.size(0) == B && K.size(1) == nh && K.size(2) == N_seq && K.size(3) == D, "K dimensions mismatch Q");
    TORCH_CHECK(V.size(0) == B && V.size(1) == nh && V.size(2) == N_seq && V.size(3) == D, "V dimensions mismatch Q");

    // Validate Bc_arg and Br_arg
    TORCH_CHECK(Bc_arg > 0, "Bc must be positive");
    TORCH_CHECK(Br_arg > 0, "Br must be positive");
    // Often, Bc and Br are multiples of 32 or 64 for warp-level efficiency.
    // TORCH_CHECK(Bc_arg % 32 == 0, "Bc_arg should preferably be a multiple of 32 for efficiency.");
    // TORCH_CHECK(Br_arg % 32 == 0, "Br_arg should preferably be a multiple of 32 for efficiency.");

    // Calculate number of blocks for sequence length (Tc and Tr)
    const int Tc = (N_seq + Bc_arg - 1) / Bc_arg; // Number of K/V blocks
    const int Tr = (N_seq + Br_arg - 1) / Br_arg; // Number of Q/O blocks

    // Calculate softmax scaling factor
    const float softmax_scale = 1.0 / std::sqrt(static_cast<float>(D));

    // 2. Allocate output tensors on the GPU
    torch::Tensor O = torch::zeros_like(Q); // Output tensor
    torch::Tensor l = torch::zeros({B, nh, N_seq}, Q.options()); // Running sum_exp_val (l_i)
    torch::Tensor m = torch::full({B, nh, N_seq}, -INFINITY, Q.options()); // Running max_val (m_i)

    // Get raw pointers to tensor data
    float *d_Q = Q.data_ptr<float>();
    float *d_K = K.data_ptr<float>();
    float *d_V = V.data_ptr<float>();
    float *d_O = O.data_ptr<float>();
    float *d_l = l.data_ptr<float>();
    float *d_m = m.data_ptr<float>();

    // Calculate Shared Memory (SRAM) size needed per thread block
    // Shared memory layout: Qi | Kj | Vj | S
    const int KV_TILE_SIZE = Bc_arg * D; // Size of Kj or Vj in floats
    const int Q_TILE_SIZE = Br_arg * D;  // Size of Qi in floats
    const int S_TILE_SIZE = Br_arg * Bc_arg; // Size of S in floats
    const int sram_size_bytes = (Q_TILE_SIZE + 2 * KV_TILE_SIZE + S_TILE_SIZE) * sizeof(float);

    // Get maximum shared memory available on the device
    int max_sram_size_per_block;
    cudaDeviceGetAttribute(&max_sram_size_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    std::cout << "max_sram_size_per_block: " << max_sram_size_per_block << std::endl;

    // Check if requested shared memory exceeds device limits
    if (sram_size_bytes > max_sram_size_per_block) {
        TORCH_CHECK(false, "Requested shared memory (" + std::to_string(sram_size_bytes) +
                           " bytes) exceeds maximum available (" + std::to_string(max_sram_size_per_block) +
                           " bytes) per block. Consider reducing Bc or Br, or increasing N_seq, D to fit.");
    }

    // Define Grid and Block dimensions for kernel launch
    dim3 grid_dim(B, nh);       // Grid: (Batch_size, Num_heads)
    dim3 block_dim(Bc_arg);     // Block: (Br_arg threads per block)
                                // Each thread will compute a row of Qi or O, up to Br rows.
                                // If tx < Br is checked, all threads in block_dim.x must be <= Br.
                                // Setting block_dim.x = Br_arg ensures all threads are utilized for the 'if (tx < Br)' path.

    // Launch the CUDA kernel
    flash_attn_v1_kernel<<<grid_dim, block_dim, sram_size_bytes>>>(
        d_Q, d_K, d_V, N_seq, D, Tc, Tr, Bc_arg, Br_arg, softmax_scale, d_l, d_m, d_O);

    // Synchronize CUDA device to ensure kernel completion before returning
    cudaDeviceSynchronize();

    return O; // Return the computed output tensor
}