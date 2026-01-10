import torch
import triton
import triton.language as tl
import itertools

# -----------------------------------------------------------------------------
# TRITON FWD KERNEL: Unified (Entropy Optional)
# -----------------------------------------------------------------------------
@triton.jit
def _fwd_kernel(
    Q, K, V,
    sm_scale,
    L, O, Ent,
    stride_qz, stride_qm, stride_qd,
    stride_kz, stride_kn, stride_kd,
    stride_vz, stride_vn, stride_vd,
    stride_lz, stride_lm,
    stride_oz, stride_om, stride_od,
    stride_ez, stride_em,
    Z, N_Q, N_K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ENABLE_ENTROPY: tl.constexpr  # <--- Compilation Flag
):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < N_Q

    # Init accumulators
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), tl.float32)
    
    # --- CONDITIONAL COMPILE: Entropy Accumulator ---
    # If ENABLE_ENTROPY is False, this variable is optimized out entirely
    acc_s = tl.zeros((BLOCK_M,), tl.float32) if ENABLE_ENTROPY else None

    # Load Q
    q_ptrs = (
        Q + pid_z * stride_qz + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # Loop over K, V
    for start_n in range(0, N_K, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_K

        # Implicit Transpose Logic (Your validated method)
        k_ptrs = (
            K + pid_z * stride_kz + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)

        v_ptrs = (
            V + pid_z * stride_vz + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Compute Attention Scores
        qk = tl.dot(q, k)
        qk *= sm_scale

        m_i_new = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_i_new)

        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc *= alpha[:, None]
        acc += tl.dot(p, v)

        # --- CONDITIONAL COMPILE: Entropy Update ---
        if ENABLE_ENTROPY:
            term_s = tl.sum(p * qk, 1)
            acc_s = acc_s * alpha + term_s

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Finalize Output
    acc = acc / l_i[:, None]
    
    # Store Output
    o_ptrs = (
        O + pid_z * stride_oz + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])

    # Finalize Meta-Data (L and Entropy)
    logsumexp = m_i + tl.log(l_i)
    l_ptrs = L + pid_z * stride_lz + offs_m * stride_lm
    tl.store(l_ptrs, logsumexp.to(L.dtype.element_ty), mask=mask_m)

    # --- CONDITIONAL COMPILE: Final Entropy Math & Store ---
    if ENABLE_ENTROPY:
        expected_s = acc_s / l_i
        entropy_val = logsumexp - expected_s
        
        e_ptrs = Ent + pid_z * stride_ez + offs_m * stride_em
        tl.store(e_ptrs, entropy_val.to(Ent.dtype.element_ty), mask=mask_m)

# -----------------------------------------------------------------------------
# PYTHON WRAPPER
# -----------------------------------------------------------------------------
def flash_attention_entropy(q, k, v, scale, 
        BLOCK_M=32, BLOCK_N=32, NUM_WARPS=4, NUM_STAGES=2,
        use_entropy=True
    ):
    """
    If use_entropy=False:
        - Skips entropy calculation overhead.
        - Returns (output, None, L)
    """
    B, H, N_Q, D = q.shape
    _, _, N_K, _ = k.shape
    Z = B * H
    
    q_ = q.view(Z, N_Q, D)
    k_ = k.view(Z, N_K, D)
    v_ = v.view(Z, N_K, D)

    o_ = torch.empty((Z, N_Q, D), device=q.device, dtype=q.dtype)
    L_ = torch.empty((Z, N_Q), device=q.device, dtype=torch.float32)
    
    # Conditional Allocation
    if use_entropy:
        ent_ = torch.empty((Z, N_Q), device=q.device, dtype=torch.float32)
    else:
        # We pass the output buffer as a dummy if entropy is disabled
        # (Triton pointer arithmetic needs a valid pointer base, even if unused)
        ent_ = o_ 

    grid = (triton.cdiv(N_Q, BLOCK_M), Z)

    _fwd_kernel[grid](
        q_, k_, v_, scale, L_, o_, ent_,
        q_.stride(0), q_.stride(1), q_.stride(2),
        k_.stride(0), k_.stride(1), k_.stride(2),
        v_.stride(0), v_.stride(1), v_.stride(2),
        L_.stride(0), L_.stride(1),
        o_.stride(0), o_.stride(1), o_.stride(2),
        ent_.stride(0), ent_.stride(1),
        Z, N_Q, N_K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        ENABLE_ENTROPY=use_entropy, # <--- Passes the flag to JIT
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )

    o = o_.view(B, H, N_Q, D)
    L = L_.view(B, H, N_Q)
    
    if use_entropy:
        ent = ent_.view(B, H, N_Q)
        return o, -ent, L
    else:
        return o, None, L

# -----------------------------------------------------------------------------
# 3. NAIVE REFERENCE
# -----------------------------------------------------------------------------
def naive_attn_entropy(query, key, value, scale):
    attn = query.mul(scale) @ key.transpose(-2, -1)   # [B,H,N_Q,N_K]
    attn_weights = attn.softmax(dim=-1)
    
    # entropy = - Σ p log p   (natural log)
    log_p = torch.log(attn_weights + 1e-9)
    entropy = -(attn_weights * log_p).sum(dim=-1)     # [B,H,N_Q]
    
    output = attn_weights @ value                     # [B,H,N_Q,D]
    return output, -entropy

# -----------------------------------------------------------------------------
# 4. BENCHMARK UTILITIES
# -----------------------------------------------------------------------------
def benchmark_latency_sum(scales_data, scale, params, device, dtype):
    """Runs benchmark across multiple scales and returns SUM of latencies."""
    
    total_latency = 0.0
    total_flops = 0.0
    
    # Iterate through the 4 scales (32, 40, 48, 64)
    for (N_Q, N_K) in scales_data:
        # 1. Allocate Tensors for this specific scale
        # We re-allocate inside the loop to simulate the changing shapes
        # (Overhead of allocation is NOT measured by do_bench)
        q = torch.randn((BATCH, HEADS, N_Q, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((BATCH, HEADS, N_K, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((BATCH, HEADS, N_K, D_HEAD), device=device, dtype=dtype)
        
        # 2. Compilation Check (First run)
        torch.cuda.empty_cache()
        try:
            flash_attention_entropy(q, k, v, scale, **params)
        except triton.runtime.autotuner.OutOfResources:
            return None, "OOM/Reg Spill"
        except Exception as e:
            return None, f"Error: {e}"
        
        # 3. Timing
        fn = lambda: flash_attention_entropy(q, k, v, scale, **params)
        try:
            # We use fewer reps to save time since we run 4x per config
            ms = triton.testing.do_bench(fn, warmup=5, rep=20) 
            total_latency += ms
            
            # Add FLOPS for this specific scale
            # 4 * B * H * N_Q * N_K * D
            scale_flops = 4 * BATCH * HEADS * N_Q * N_K * D_HEAD
            total_flops += scale_flops
            
        except Exception as e:
            return None, f"Runtime Error: {e}"

    return total_latency, total_flops

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    
    # ==========================================
    # STEP 1: VERIFICATION (Small Tensors)
    # ==========================================
    print(f"--- 1. Verifying Correctness ---")
    BATCH_V, HEADS_V = 2, 4
    N_Q_V, N_K_V, D_HEAD_V = 256, 1024, 64
    
    q_v = torch.randn((BATCH_V, HEADS_V, N_Q_V, D_HEAD_V), device=device, dtype=dtype)
    k_v = torch.randn((BATCH_V, HEADS_V, N_K_V, D_HEAD_V), device=device, dtype=dtype)
    v_v = torch.randn((BATCH_V, HEADS_V, N_K_V, D_HEAD_V), device=device, dtype=dtype)
    scale_v = 1.0 / (D_HEAD_V ** 0.5)

    out_triton, ent_triton, _ = flash_attention_entropy(q_v, k_v, v_v, scale_v)
    out_ref, ent_ref = naive_attn_entropy(q_v, k_v, v_v, scale_v)

    out_ref = out_ref.to(out_triton.dtype)
    ent_ref = ent_ref.to(ent_triton.dtype)

    diff_out = torch.max(torch.abs(out_triton - out_ref))
    diff_ent = torch.max(torch.abs(ent_triton - ent_ref))

    print(f"Output Max Diff:  {diff_out}")
    print(f"Entropy Max Diff: {diff_ent}")

    if torch.allclose(out_triton, out_ref, atol=1e-2) and \
       torch.allclose(ent_triton, ent_ref, atol=1e-2):
        print("✅ Implementation Matches (cross-attention)!")
    else:
        print("❌ Mismatch found. Aborting benchmark.")
        exit()

    # ==========================================
    # STEP 2: MULTI-SCALE BENCHMARK (RTX 3090)
    # ==========================================
    print(f"\n--- 2. Benchmarking Total Latency (Last 4 Scales) ---")
    BATCH = 2
    HEADS = 16
    D_HEAD = 128
    
    # --- Define the Last 4 Scales ---
    # Scales: 32, 40, 48, 64
    # N_Q = Scale^2
    # N_K = Cumulative sum of all previous scales squared
    # Calculated manually based on [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
    
    scales_data = [
        # (N_Q, N_K)
        (32*32, 2521),   # Scale 32
        (40*40, 4121),   # Scale 40
        (48*48, 6425),   # Scale 48
        (64*64, 10521)   # Scale 64 (Last one)
    ]
    
    scale_factor = 1.0 / (D_HEAD ** 0.5)
    
    print(f"Testing Shapes: {scales_data}")
    print(f"Goal: Minimize SUM of latencies for these 4 steps.")
    print("-" * 115)
    print(f"{'BLOCK_M':<8} | {'BLOCK_N':<8} | {'WARPS':<6} | {'STAGES':<6} | {'Total Time (ms)':<16} | {'Effective TFLOPS':<16}")
    print("-" * 115)

    # 3090 Tuning Grid (Optimized for D=128)
    configs = [
        # Baseline (Conservative)
        (64, 32, 4, 2),
        
        # Larger M (Good for query loading)
        (64, 64, 4, 2),
        (128, 32, 4, 2),
        
        # --- The "High Pipeline" Candidates for D=128 ---
        # Reducing N to 32 allows us to increase STAGES without OOM
        (64, 32, 4, 3),  
        (64, 32, 4, 4),  # <--- Expected Winner on 3090
        (64, 32, 8, 3),  
        
        # Small tiles for high occupancy
        (32, 32, 4, 2),

        # --- Strategy 1: Increase Pipeline Depth (Hide Latency) ---
        # Small blocks use very little SRAM, so we can afford deep stages.
        (32, 32, 4, 3),
        (32, 32, 4, 4),
        (32, 32, 4, 5),  # Aggressive pipelining

        # --- Strategy 2: Increase Block N (Better Memory Coalescing) ---
        # Loading 32 elements is okay, but 64 is better bandwidth-wise.
        # Since M is small (32), M*N accumulator registers won't explode.
        (32, 64, 4, 2),
        (32, 64, 4, 3),  
        (32, 64, 4, 4),

        # --- Strategy 3: Aggressive Key Loading ---
        # Can we load huge chunks of K? (Risk of SRAM OOM, but worth a shot)
        (32, 128, 4, 2),
        (32, 128, 4, 3),

        # --- Strategy 4: The "Tiny" Competitor (Low Warps) ---
        # For small 32x32 tiles, maybe 4 warps (128 threads) is too much sync?
        # Try 2 warps (64 threads) if Triton allows.
        (32, 32, 2, 2),
        (32, 64, 2, 2),
        
        # --- Re-check Previous "Runner Up" ---
        # 128x32 was close (51ms). Let's see if pipelining helps IT.
        (128, 32, 4, 3),
    ]

    best_total_ms = float('inf')
    best_config = None

    for (bm, bn, nw, ns) in configs:
        params = {"BLOCK_M": bm, "BLOCK_N": bn, "NUM_WARPS": nw, "NUM_STAGES": ns}
        
        # Run sum benchmark
        res = benchmark_latency_sum(scales_data, scale_factor, params, device, dtype)
        
        if res[0] is not None:
            total_ms, total_flops = res
            
            # TFLOPS = (Total FLOPs / 1e12) / (Total Time / 1000)
            tflops = (total_flops / 1e12) / (total_ms / 1000)
            
            print(f"{bm:<8} | {bn:<8} | {nw:<6} | {ns:<6} | {total_ms:<16.3f} | {tflops:<16.2f}")
            
            if total_ms < best_total_ms:
                best_total_ms = total_ms
                best_config = params
        else:
            error_msg = res[1]
            print(f"{bm:<8} | {bn:<8} | {nw:<6} | {ns:<6} | {'FAILED':<16} | {error_msg}")

    print("-" * 115)
    if best_config:
        print(f"🚀 Best Configuration: {best_config}")
        print(f"⏱️  Total Latency (Sum of 4 scales): {best_total_ms:.3f} ms")