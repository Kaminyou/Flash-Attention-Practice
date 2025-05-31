#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void flash_attn_v1_kernel(const float *Q,
                                     const float *K,
                                     const float *V,
                                     const int N,
                                     const int d,
                                     const int Tc,
                                     const int Tr,
                                     const int Bc,
                                     const int Br,
                                     const float softmax_scale,
                                     float *l,
                                     float *m,
                                     float *O)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    const int KV_TILE_SIZE = Bc * d; // size of Kj, Vj
    const int Q_TILE_SIZE = Br * d;  // size of Qi
    // const int S_TILE_SIZE = Br * Bc; // size of Sij = softmax(Qi * Kj^T * softmax_scale)
    float *Qi = sram;
    float *Kj = &sram[Q_TILE_SIZE];
    float *Vj = &sram[Q_TILE_SIZE + KV_TILE_SIZE];
    float *S = &sram[Q_TILE_SIZE + KV_TILE_SIZE * 2];

    // outer loop
    for (int j = 0; j < Tc; j++)
    {
        // Load Kj, Vj from HBM to SRAM
        for (int x = 0; x < d; x++)
        {
            Kj[(tx * d) + x] = K[qkv_offset + (KV_TILE_SIZE * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (KV_TILE_SIZE * j) + (tx * d) + x];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)
        {
            if (tx < Br)
            {
                // Load Qi to SRAM, l and m to registers
                for (int x = 0; x < d; x++)
                {
                    Qi[(tx * d) + x] = Q[qkv_offset + (Q_TILE_SIZE * i) + (tx * d) + x];
                }
                float row_m_prev = m[lm_offset + (Br * i) + tx];
                float row_l_prev = l[lm_offset + (Br * i) + tx];

                // S = QK^T, row_m = rowmax(S)
                float row_m = -INFINITY;
                for (int y = 0; y < Bc; y++)
                {
                    float sum = 0;
                    for (int x = 0; x < d; x++)
                    {
                        sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                    }
                    sum *= softmax_scale;
                    S[(Bc * tx) + y] = sum;

                    if (sum > row_m)
                        row_m = sum;
                }

                // P = exp(S - row_m), row_l = rowsum(P)
                float row_l = 0;
                for (int y = 0; y < Bc; y++)
                {
                    S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                    row_l += S[(Bc * tx) + y];
                }

                // Compute new m and l
                float row_m_new = max(row_m_prev, row_m);
                float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

                // Write O, l, m to HBM
                for (int x = 0; x < d; x++)
                {
                    float pv = 0; // Pij * Vj
                    for (int y = 0; y < Bc; y++)
                    {
                        pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                    }
                    O[qkv_offset + (Q_TILE_SIZE * i) + (tx * d) + x] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (Q_TILE_SIZE * i) + (tx * d) + x]) + (__expf(row_m - row_m_new) * pv));
                }
                m[lm_offset + (Br * i) + tx] = row_m_new;
                l[lm_offset + (Br * i) + tx] = row_l_new;
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

    // Check if requested shared memory exceeds device limits
    if (sram_size_bytes > max_sram_size_per_block) {
        TORCH_CHECK(false, "Requested shared memory (" + std::to_string(sram_size_bytes) +
                           " bytes) exceeds maximum available (" + std::to_string(max_sram_size_per_block) +
                           " bytes) per block. Consider reducing Bc or Br, or increasing N_seq, D to fit.");
    }

    // Define Grid and Block dimensions for kernel launch
    dim3 grid_dim(B, nh);       // Grid: (Batch_size, Num_heads)
    dim3 block_dim(Br_arg);     // Block: (Br_arg threads per block)
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