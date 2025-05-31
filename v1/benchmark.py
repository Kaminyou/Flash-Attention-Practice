import argparse
import gc
import math

import torch
import flash_attn_v1_ext


def argument():
    parser = argparse.ArgumentParser('Benchmark settings')
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=4,
        help='Batch size',
    )
    parser.add_argument(
        '-nh',
        '--head',
        type=int,
        default=8,
        help='Head number',
    )
    parser.add_argument(
        '-n',
        '--length',
        type=int,
        default=128,
        help='Sequence length',
    )
    parser.add_argument(
        '-d',
        '--dimension',
        type=int,
        default=64,
        help='Dimension',
    )
    parser.add_argument(
        '--bc',
        type=int,
        default=32,
        help='Bc',
    )
    parser.add_argument(
        '--br',
        type=int,
        default=16,
        help='Br',
    )
    args = parser.parse_args()
    return args


def calculate_pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


def main():
    args = argument()
    B = args.batch_size
    nh = args.head
    N = args.length
    D = args.dimension
    Bc = args.bc
    Br = args.br

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a CUDA-enabled GPU.")
        return

    device = 'cuda'
    dtype = torch.float32 # Your C++ kernel uses float32

    print(f"Testing FlashAttention with parameters: B={B}, nh={nh}, N={N}, D={D}, Bc={Bc}, Br={Br}")
    print(f"Device: {device}, Dtype: {dtype}")

    # Generate random input tensors on the GPU
    q = torch.randn(B, nh, N, D, dtype=dtype, device=device)
    k = torch.randn(B, nh, N, D, dtype=dtype, device=device)
    v = torch.randn(B, nh, N, D, dtype=dtype, device=device)

    # Clear CUDA memory cache and ensure no residual allocations
    torch.cuda.empty_cache()
    gc.collect()

    # --- 1. Compute output using your custom FlashAttention kernel ---
    print("\n--- Running custom FlashAttention kernel ---")
    try:
        q_cont = q.contiguous()
        k_cont = k.contiguous()
        v_cont = v.contiguous()

        softmax_scale_kernel = 1.0 / math.sqrt(D)

        # Reset memory stats before benchmarking
        torch.cuda.reset_peak_memory_stats(device=device)
        initial_mem_custom = torch.cuda.memory_allocated(device=device)

        start_event_custom = torch.cuda.Event(enable_timing=True)
        end_event_custom = torch.cuda.Event(enable_timing=True)

        start_event_custom.record()
        _ = flash_attn_v1_ext.forward(q_cont, k_cont, v_cont, Bc, Br)
        end_event_custom.record()
        torch.cuda.synchronize()

        custom_time_ms = start_event_custom.elapsed_time(end_event_custom)
        peak_mem_custom = torch.cuda.max_memory_allocated(device=device)
        final_mem_custom = torch.cuda.memory_allocated(device=device)

        print(f"Custom FlashAttention time: {custom_time_ms:.4f} ms")
        print(f"Custom FlashAttention VRAM (allocated before run): {initial_mem_custom / (1024**2):.2f} MB")
        print(f"Custom FlashAttention VRAM (peak during run): {peak_mem_custom / (1024**2):.2f} MB")
        print(f"Custom FlashAttention VRAM (allocated after run): {final_mem_custom / (1024**2):.2f} MB")

    except Exception as e:
        print(f"Error running custom FlashAttention: {e}")
        return

    torch.cuda.empty_cache()
    gc.collect()

    print("\n--- Running standard PyTorch attention ---")
    softmax_scale_pytorch = 1.0 / math.sqrt(D)

    # Reset memory stats before benchmarking
    torch.cuda.reset_peak_memory_stats(device=device)
    initial_mem_pytorch = torch.cuda.memory_allocated(device=device)

    start_event_pytorch = torch.cuda.Event(enable_timing=True)
    end_event_pytorch = torch.cuda.Event(enable_timing=True)

    start_event_pytorch.record()
    output_pytorch = calculate_pytorch_attention(q, k, v, softmax_scale_pytorch)
    end_event_pytorch.record()
    torch.cuda.synchronize()

    pytorch_time_ms = start_event_pytorch.elapsed_time(end_event_pytorch)
    peak_mem_pytorch = torch.cuda.max_memory_allocated(device=device)
    final_mem_pytorch = torch.cuda.memory_allocated(device=device)

    print(f"Standard PyTorch attention time: {pytorch_time_ms:.4f} ms")
    print(f"Standard PyTorch attention VRAM (allocated before run): {initial_mem_pytorch / (1024**2):.2f} MB")
    print(f"Standard PyTorch attention VRAM (peak during run): {peak_mem_pytorch / (1024**2):.2f} MB")
    print(f"Standard PyTorch attention VRAM (allocated after run): {final_mem_pytorch / (1024**2):.2f} MB")

    print(f"\n--- Summary ---")
    print(f"Performance comparison: Custom FlashAttention was {pytorch_time_ms / custom_time_ms:.2f}x faster than standard PyTorch attention.")
    print(f"VRAM Peak comparison: Standard PyTorch attention used {peak_mem_pytorch / (1024**2):.2f} MB (peak), Custom FlashAttention used {peak_mem_custom / (1024**2):.2f} MB (peak).")
    print(f"FlashAttention saved {(peak_mem_pytorch - peak_mem_custom) / (1024**2):.2f} MB of VRAM (or {(1 - peak_mem_custom / peak_mem_pytorch) * 100:.2f}% reduction).")


if __name__ == '__main__':
    main()
