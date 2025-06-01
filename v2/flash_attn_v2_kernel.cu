#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <iostream>


__global__ void flash_attn_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int BrDim, int BcDim, int Bs,
    int Nh, int N, int dim)
{
    extern __shared__ float shared_mem[];
    float* Qs = shared_mem;
    float* Ks = Qs + BrDim * dim;
    float* Ss = Ks + BcDim * dim;
    float* Vs = Ss + BrDim * BcDim;
    float* Os = Vs + BcDim * dim;
    float* ls = Os + BrDim * dim;
    float* ms = ls + BrDim;
    float* expMaxDelta = ms + BrDim;

    const int BrIdx = blockIdx.x;  // Row block index
    const int NhIdx = blockIdx.y;  // Head index
    const int BsIdx = blockIdx.z;  // Batch index

    const int TcIdx = threadIdx.x;
    const int TrIdx = threadIdx.y;
    const int TcDim = blockDim.x;
    const int TrDim = blockDim.y;

    const int global_offset = (BsIdx * Nh + NhIdx) * N * dim;

    // Initialize ms and ls
    for (int i = TrIdx * TcDim + TcIdx; i < BrDim; i += TrDim * TcDim) {
        ms[i] = -INFINITY;
        ls[i] = 0.0f;
    }

    // Initialize Os
    for (int i = TrIdx; i < BrDim; i += TrDim) {
        for (int j = TcIdx; j < dim; j += TcDim) {
            Os[i * dim + j] = 0.0f;
        }
    }

    __syncthreads();

    // Load Qs
    for (int i = TrIdx; i < BrDim; i += TrDim) {
        for (int j = TcIdx; j < dim; j += TcDim) {
            int idx = BrIdx * BrDim + i;
            if (idx < N) {
                Qs[i * dim + j] = Q[global_offset + idx * dim + j];
            }
        }
    }

    const int num_Bc = (N + BcDim - 1) / BcDim;
    const float scale = 1.0f / sqrtf(dim);

    for (int BcIdx = 0; BcIdx < num_Bc; ++BcIdx) {
        // Load Ks and Vs
        for (int i = TrIdx; i < BcDim; i += TrDim) {
            for (int j = TcIdx; j < dim; j += TcDim) {
                int idx = BcIdx * BcDim + i;
                if (idx < N) {
                    Ks[i * dim + j] = K[global_offset + idx * dim + j];
                    Vs[i * dim + j] = V[global_offset + idx * dim + j];
                }
            }
        }
        __syncthreads();

        // Compute scaled dot-product attention scores
        for (int i = TrIdx; i < BrDim; i += TrDim) {
            for (int j = TcIdx; j < BcDim; j += TcDim) {
                if ((BrIdx * BrDim + i < N) && (BcIdx * BcDim + j < N)) {
                    float score = 0.0f;
                    for (int k = 0; k < dim; ++k) {
                        score += Qs[i * dim + k] * Ks[j * dim + k];
                    }
                    Ss[i * BcDim + j] = score * scale;
                }
            }
        }

        __syncthreads();

        // Softmax (max subtraction)
        for (int i = TrIdx; i < BrDim; i += TrDim) {
            if (BrIdx * BrDim + i >= N) continue;

            float row_max = -INFINITY;
            for (int j = TcIdx; j < BcDim; j += TcDim) {
                if (BcIdx * BcDim + j < N) {
                    row_max = fmaxf(row_max, Ss[i * BcDim + j]);
                }
            }
            for (int stride = 16; stride > 0; stride /= 2) {
                row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, stride));
            }

            if (TcIdx == 0) {
                row_max = fmaxf(ms[i], row_max);
            }
            row_max = __shfl_sync(0xffffffff, row_max, 0);

            for (int j = TcIdx; j < BcDim; j += TcDim) {
                if (BcIdx * BcDim + j < N) {
                    Ss[i * BcDim + j] = expf(Ss[i * BcDim + j] - row_max);
                }
            }

            if (TcIdx == 0) {
                expMaxDelta[i] = expf(ms[i] - row_max);
                ms[i] = row_max;
            }
        }
        __syncthreads();

        // Sum and multiply with Vs
        for (int i = TrIdx; i < BrDim; i += TrDim) {
            if (BrIdx * BrDim + i >= N) continue;

            float row_sum = 0.0f;
            for (int j = TcIdx; j < BcDim; j += TcDim) {
                if (BcIdx * BcDim + j < N) {
                    row_sum += Ss[i * BcDim + j];
                }
            }

            for (int stride = 16; stride > 0; stride /= 2) {
                row_sum += __shfl_down_sync(0xffffffff, row_sum, stride);
            }

            if (TcIdx == 0) {
                ls[i] = fmaxf(ls[i] * expMaxDelta[i] + row_sum, 1e-7f);
            }

            for (int j = TcIdx; j < dim; j += TcDim) {
                float val = 0.0f;
                for (int k = 0; k < BcDim; ++k) {
                    if (BcIdx * BcDim + k < N) {
                        val += Ss[i * BcDim + k] * Vs[k * dim + j];
                    }
                }
                Os[i * dim + j] = Os[i * dim + j] * expMaxDelta[i] + val;
            }
        }
        __syncthreads();
    }

    // Final normalization
    for (int i = TrIdx; i < BrDim; i += TrDim) {
        if (BrIdx * BrDim + i < N) {
            for (int j = TcIdx; j < dim; j += TcDim) {
                Os[i * dim + j] /= ls[i];
            }
            if (TcIdx == 0) {
                ls[i] = ms[i] + logf(ls[i]);
            }
        }
    }
    __syncthreads();
    // Write to global memory
    for (int i = TrIdx; i < BrDim; i += TrDim) {
        int idx = BrIdx * BrDim + i;
        if (idx < N) {
            for (int j = TcIdx; j < dim; j += TcDim) {
                O[global_offset + idx * dim + j] = Os[i * dim + j];
            }
            if (TcIdx == 0) {
                L[BsIdx * Nh * N + NhIdx * N + idx] = ls[i];
            }
        }
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int Bc_arg, int Br_arg) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

    int Bs = Q.size(0);
    int Nh = Q.size(1);
    int N  = Q.size(2);
    int dim = Q.size(3);
    int BrDim = Br_arg;
    int BcDim = Bc_arg;

    auto O = torch::empty_like(Q);
    auto L = torch::empty({Bs, Nh, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 blockDim(32, 4);  // Tune as needed
    dim3 gridDim((N + BrDim - 1) / BrDim, Nh, Bs);

    size_t shmem_size =
        sizeof(float) * (BrDim * dim + BcDim * dim + BcDim * dim + BrDim * dim) +  // Qs, Ks, Vs, Os
        sizeof(float) * (BrDim * BcDim + 3 * BrDim);                              // Ss, ls, ms, expMaxDelta

    flash_attn_v2_kernel<<<gridDim, blockDim, shmem_size>>>(
        reinterpret_cast<const float *>(Q.data_ptr<float>()),
        reinterpret_cast<const float *>(K.data_ptr<float>()),
        reinterpret_cast<const float *>(V.data_ptr<float>()),
        reinterpret_cast<float *>(O.data_ptr<float>()),
        L.data_ptr<float>(),
        BrDim, BcDim,
        Bs, Nh, N, dim
    );

    return O;
}