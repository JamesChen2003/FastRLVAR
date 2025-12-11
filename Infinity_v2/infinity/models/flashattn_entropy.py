import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# TRITON FWD KERNEL: FlashAttention + Entropy
# Q shape: [Z, N_Q, D],  K,V shape: [Z, N_K, D]  where Z = B * H
# -----------------------------------------------------------------------------
@triton.jit
def _fwd_kernel(
    Q, K, V,
    sm_scale,
    L, O, Ent,                      # outputs: L(z, m), O(z, m, d), Ent(z, m)
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
):
    pid_m = tl.program_id(0)  # row-tile id over queries
    pid_z = tl.program_id(1)  # batch*head id

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query indices
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < N_Q

    # ------------------------------
    # Init running stats (Algo line 5)
    # ------------------------------
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), tl.float32)
    acc_s = tl.zeros((BLOCK_M,), tl.float32)  # Σ exp(S - m) * S

    # ------------------------------
    # Load Q block once (Algo line 4)
    # ------------------------------
    q_ptrs = (
        Q
        + pid_z * stride_qz
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # ------------------------------
    # Loop over K/V blocks (Algo line 6–11) along key length N_K
    # ------------------------------
    for start_n in range(0, N_K, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_K

        # K tile: [D, N]
        k_ptrs = (
            K
            + pid_z * stride_kz
            + offs_n[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)

        # V tile: [N, D]
        v_ptrs = (
            V
            + pid_z * stride_vz
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # 1. S = Q K^T  (scores) -> [M, N]
        qk = tl.dot(q, k)
        qk *= sm_scale

        # 2. Update running max m_i
        m_i_new = tl.max(qk, 1)                 # rowmax over N
        m_i_new = tl.maximum(m_i, m_i_new)

        # 3. Compute exp scores with new max
        alpha = tl.exp(m_i - m_i_new)           # [M]
        p = tl.exp(qk - m_i_new[:, None])       # [M, N]

        # 4. Update output accumulator
        acc *= alpha[:, None]
        acc += tl.dot(p, v)

        # 5. Update entropy accumulator: Σ exp(S - m) * S
        term_s = tl.sum(p * qk, 1)              # [M]
        acc_s = acc_s * alpha + term_s

        # 6. Update normalizer ℓ_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # ------------------------------
    # Finalize
    # ------------------------------
    # O = acc / ℓ
    acc = acc / l_i[:, None]

    # logsumexp = m_i + log(ℓ_i)
    logsumexp = m_i + tl.log(l_i)

    # E_p[S] = acc_s / ℓ_i
    expected_s = acc_s / l_i

    # Entropy H = log Z - E_p[S] = - Σ p log p  (positive)
    entropy_val = logsumexp - expected_s

    # ------------------------------
    # Stores
    # ------------------------------
    # O(z, m, d)
    o_ptrs = (
        O
        + pid_z * stride_oz
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])

    # L(z, m) = logsumexp
    l_ptrs = L + pid_z * stride_lz + offs_m * stride_lm
    tl.store(l_ptrs, logsumexp.to(L.dtype.element_ty), mask=mask_m)

    # Ent(z, m) = H
    e_ptrs = Ent + pid_z * stride_ez + offs_m * stride_em
    tl.store(e_ptrs, entropy_val.to(Ent.dtype.element_ty), mask=mask_m)

# -----------------------------------------------------------------------------
# PYTHON WRAPPER
# -----------------------------------------------------------------------------
def flash_attention_entropy(q, k, v, scale):
    """
    q: [B, H, N_Q, D]
    k, v: [B, H, N_K, D]
    Returns:
      o:      [B, H, N_Q, D]
      ent:    [B, H, N_Q]   (positive entropy)
      L:      [B, H, N_Q]   (logsumexp over K-dim)
    """
    B, H, N_Q, D = q.shape
    _, _, N_K, _ = k.shape

    BLOCK_M = 64    # tile size over queries
    BLOCK_N = 32     # tile size over keys
    BLOCK_D = D

    # Flatten batch and heads -> Z dimension
    Z = B * H
    q_ = q.view(Z, N_Q, D)
    k_ = k.view(Z, N_K, D)
    v_ = v.view(Z, N_K, D)

    o_ = torch.empty((Z, N_Q, D), device=q.device, dtype=q.dtype)
    ent_ = torch.empty((Z, N_Q), device=q.device, dtype=torch.float32)
    L_ = torch.empty((Z, N_Q), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(N_Q, BLOCK_M), Z)

    _fwd_kernel[grid](
        q_, k_, v_,
        scale,
        L_, o_, ent_,
        q_.stride(0), q_.stride(1), q_.stride(2),
        k_.stride(0), k_.stride(1), k_.stride(2),
        v_.stride(0), v_.stride(1), v_.stride(2),
        L_.stride(0), L_.stride(1),
        o_.stride(0), o_.stride(1), o_.stride(2),
        ent_.stride(0), ent_.stride(1),
        Z, N_Q, N_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    o = o_.view(B, H, N_Q, D)
    ent = ent_.view(B, H, N_Q)
    L = L_.view(B, H, N_Q)
    return o, -ent, L

# -----------------------------------------------------------------------------
# Naive reference for checking (cross-attention)
# -----------------------------------------------------------------------------
def naive_attn_entropy(query, key, value, scale):
    # query: [B, H, N_Q, D]
    # key,value: [B, H, N_K, D]
    attn = query.mul(scale) @ key.transpose(-2, -1)   # [B,H,N_Q,N_K]
    attn_weights = attn.softmax(dim=-1)
    # entropy = - Σ p log p   (natural log)
    log_p = torch.log(attn_weights + 1e-9)
    entropy = -(attn_weights * log_p).sum(dim=-1)     # [B,H,N_Q]
    output = attn_weights @ value                     # [B,H,N_Q,D]
    return output, entropy

if __name__ == "__main__":
    torch.manual_seed(0)
    BATCH, HEADS = 2, 4
    N_Q, N_K, D_HEAD = 256, 1024, 64   # NOTE: N_Q != N_K
    dtype = torch.float16
    device = "cuda"

    q = torch.randn((BATCH, HEADS, N_Q, D_HEAD), device=device, dtype=dtype)
    k = torch.randn((BATCH, HEADS, N_K, D_HEAD), device=device, dtype=dtype)
    v = torch.randn((BATCH, HEADS, N_K, D_HEAD), device=device, dtype=dtype)
    scale = 1.0 / (D_HEAD ** 0.5)

    # Triton
    out_triton, ent_triton, _ = flash_attention_entropy(q, k, v, scale)

    # Naive reference
    out_ref, ent_ref = naive_attn_entropy(q, k, v, scale)

    # --- make dtypes match ---
    out_ref = out_ref.to(out_triton.dtype)
    ent_ref = ent_ref.to(ent_triton.dtype)

    print(f"Output Max Diff: {torch.max(torch.abs(out_triton - out_ref))}")
    print(f"Entropy Max Diff: {torch.max(torch.abs(ent_triton - ent_ref))}")

    if torch.allclose(out_triton, out_ref, atol=1e-2) and \
       torch.allclose(ent_triton, ent_ref, atol=1e-2):
        print("✅ Implementation Matches (cross-attention)!")
    else:
        print("❌ Mismatch found.")
