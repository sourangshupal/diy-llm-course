# from sympy import Q
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_fwd_kernel(
    # -------- global memory pointers --------
    Q_ptr, K_ptr, V_ptr,           # inputs Q, K, V
    O_ptr, L_ptr,                  # outputs O, logsumexp L

    # -------- stride info (for block ptr) --------
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb, stride_ln,

    # -------- sequence lengths --------
    Nq, Nk,

    # -------- scale --------
    scale,

    # -------- compile-time constants --------
    D: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Each Triton program is responsible for:
        - one batch
        - one Query block (Q_BLOCK rows)
    """

    # ------------------------------------------------
    # Triton parallel indices
    # ------------------------------------------------
    q_block_id = tl.program_id(0)     # which Q block
    batch_id   = tl.program_id(1)     # which batch

    # =================================================
    # Construct block pointers (key to Triton)
    # =================================================
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_id * stride_qb,
        shape=(Nq, D),
        strides=(stride_qn, stride_qd),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    # K / V start at row 0 and advance inside the loop
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_id * stride_kb,
        shape=(Nk, D),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(K_BLOCK, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_id * stride_vb,
        shape=(Nk, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(K_BLOCK, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_id * stride_ob,
        shape=(Nq, D),
        strides=(stride_on, stride_od),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_id * stride_lb,
        shape=(Nq, 1),
        strides=(stride_ln, 1),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, 1),
        order=(1, 0),
    )

    # =================================================
    # Load Query block
    # =================================================
    Q_i = tl.load(Q_block_ptr)  # (Q_BLOCK, D)

    # =================================================
    # FlashAttention core state (independent per Q block)
    # =================================================
    O_acc = tl.zeros((Q_BLOCK, D), dtype=tl.float32)       # unnormalized output accumulation
    L_acc = tl.zeros((Q_BLOCK, 1), dtype=tl.float32)      # softmax denominator
    M_acc = tl.full((Q_BLOCK, 1), -float("inf"), tl.float32)  # running max

    # =================================================
    # Inner loop: iterate over all K/V blocks
    # =================================================
    for k_block_id in range(tl.cdiv(Nk, K_BLOCK)):
        K_j = tl.load(K_block_ptr)   # (K_BLOCK, D)
        V_j = tl.load(V_block_ptr)   # (K_BLOCK, D)

        # ---------------------------------------------
        # S_ij = Q_i @ K_j^T / sqrt(d)
        # ---------------------------------------------
        S_ij = tl.dot(Q_i, K_j.T) * scale  # (Q_BLOCK, K_BLOCK)

        # ---------------------------------------------
        # Causal mask (compile-time if)
        # ---------------------------------------------
        if is_causal:
            q_idx = q_block_id * Q_BLOCK + tl.arange(0, Q_BLOCK)[:, None]
            k_idx = k_block_id * K_BLOCK + tl.arange(0, K_BLOCK)[None, :]
            causal_mask = q_idx >= k_idx
            S_ij = tl.where(causal_mask, S_ij, -1e6)

        # ---------------------------------------------
        # Numerically stable softmax (core of FlashAttention)
        # ---------------------------------------------
        M_block = tl.max(S_ij, axis=1, keep_dims=True)
        M_new = tl.maximum(M_acc, M_block)

        P_ij = tl.exp(S_ij - M_block)

        L_new = (
            tl.exp(M_acc - M_new) * L_acc +
            tl.exp(M_block - M_new) * tl.sum(P_ij, axis=1, keep_dims=True)
        )

        # Type alignment (Triton detail)
        P_cast = P_ij.to(V_block_ptr.type.element_ty)

        O_new = (
            tl.exp(M_acc - M_new) * O_acc +
            tl.exp(M_block - M_new) * tl.dot(P_cast, V_j)
        )

        # ---------------------------------------------
        # Update running state
        # ---------------------------------------------
        M_acc = M_new
        L_acc = L_new
        O_acc = O_new

        # ---------------------------------------------
        # Advance K/V block pointers
        # ---------------------------------------------
        K_block_ptr = K_block_ptr.advance((K_BLOCK, 0))
        V_block_ptr = V_block_ptr.advance((K_BLOCK, 0))

    # =================================================
    # Softmax normalization
    # =================================================
    O_i = O_acc / L_acc
    L_i = M_acc + tl.log(L_acc)

    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)


@triton.jit
def flash_attention_bwd_dkdv_kernel(
    # -------- global memory pointers --------
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Dsum_ptr,
    dK_ptr, dV_ptr,

    # -------- stride info --------
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,   # dO strides
    stride_lb, stride_ln,
    stride_db, stride_dn,              # Dsum strides
    stride_dkb, stride_dkn, stride_dkd,
    stride_dvb, stride_dvn, stride_dvd,

    Nq, Nk,
    scale,

    D: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Each Triton program computes dK/dV for one (batch, K/V block), looping
    over all Q blocks that can see it (all of them, unless is_causal).
    """
    k_block_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_id * stride_kb,
        shape=(Nk, D), strides=(stride_kn, stride_kd),
        offsets=(k_block_id * K_BLOCK, 0),
        block_shape=(K_BLOCK, D), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_id * stride_vb,
        shape=(Nk, D), strides=(stride_vn, stride_vd),
        offsets=(k_block_id * K_BLOCK, 0),
        block_shape=(K_BLOCK, D), order=(1, 0),
    )
    K_j = tl.load(K_block_ptr)
    V_j = tl.load(V_block_ptr)

    dK_acc = tl.zeros((K_BLOCK, D), dtype=tl.float32)
    dV_acc = tl.zeros((K_BLOCK, D), dtype=tl.float32)

    num_q_blocks = Nq // Q_BLOCK
    # Causal: a Q block entirely before this K block contributes nothing
    # (all its rows are masked out), so skip straight to the diagonal.
    start_q_block = k_block_id if is_causal else 0

    for q_block_id in range(start_q_block, num_q_blocks):
        Q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + batch_id * stride_qb,
            shape=(Nq, D), strides=(stride_qn, stride_qd),
            offsets=(q_block_id * Q_BLOCK, 0),
            block_shape=(Q_BLOCK, D), order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            base=dO_ptr + batch_id * stride_ob,
            shape=(Nq, D), strides=(stride_on, stride_od),
            offsets=(q_block_id * Q_BLOCK, 0),
            block_shape=(Q_BLOCK, D), order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            base=L_ptr + batch_id * stride_lb,
            shape=(Nq, 1), strides=(stride_ln, 1),
            offsets=(q_block_id * Q_BLOCK, 0),
            block_shape=(Q_BLOCK, 1), order=(1, 0),
        )
        D_block_ptr = tl.make_block_ptr(
            base=Dsum_ptr + batch_id * stride_db,
            shape=(Nq, 1), strides=(stride_dn, 1),
            offsets=(q_block_id * Q_BLOCK, 0),
            block_shape=(Q_BLOCK, 1), order=(1, 0),
        )

        Q_i = tl.load(Q_block_ptr)
        dO_i = tl.load(dO_block_ptr)
        L_i = tl.load(L_block_ptr)
        Di = tl.load(D_block_ptr)

        # Recompute P_ij = exp(S_ij - L_i); L_i already encodes (m_i + log l_i)
        # from the forward pass, so this reproduces the exact softmax weights.
        S_ij = tl.dot(Q_i, K_j.T) * scale

        if is_causal:
            q_idx = q_block_id * Q_BLOCK + tl.arange(0, Q_BLOCK)[:, None]
            k_idx = k_block_id * K_BLOCK + tl.arange(0, K_BLOCK)[None, :]
            causal_mask = q_idx >= k_idx
            P_ij = tl.where(causal_mask, tl.exp(S_ij - L_i), 0.0)
        else:
            P_ij = tl.exp(S_ij - L_i)

        P_cast = P_ij.to(dO_i.dtype)
        dV_acc += tl.dot(tl.trans(P_cast), dO_i)

        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - Di)
        dS_cast = dS_ij.to(Q_i.dtype)

        dK_acc += tl.dot(tl.trans(dS_cast), Q_i) * scale

    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_id * stride_dkb,
        shape=(Nk, D), strides=(stride_dkn, stride_dkd),
        offsets=(k_block_id * K_BLOCK, 0),
        block_shape=(K_BLOCK, D), order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_id * stride_dvb,
        shape=(Nk, D), strides=(stride_dvn, stride_dvd),
        offsets=(k_block_id * K_BLOCK, 0),
        block_shape=(K_BLOCK, D), order=(1, 0),
    )
    tl.store(dK_block_ptr, dK_acc.to(K_j.dtype))
    tl.store(dV_block_ptr, dV_acc.to(V_j.dtype))


@triton.jit
def flash_attention_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Dsum_ptr,
    dQ_ptr,

    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb, stride_ln,
    stride_db, stride_dn,
    stride_dqb, stride_dqn, stride_dqd,

    Nq, Nk,
    scale,

    D: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Each Triton program computes dQ for one (batch, Q block), looping over
    all K/V blocks it attends to.
    """
    q_block_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_id * stride_qb,
        shape=(Nq, D), strides=(stride_qn, stride_qd),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D), order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_id * stride_ob,
        shape=(Nq, D), strides=(stride_on, stride_od),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D), order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_id * stride_lb,
        shape=(Nq, 1), strides=(stride_ln, 1),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, 1), order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        base=Dsum_ptr + batch_id * stride_db,
        shape=(Nq, 1), strides=(stride_dn, 1),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, 1), order=(1, 0),
    )

    Q_i = tl.load(Q_block_ptr)
    dO_i = tl.load(dO_block_ptr)
    L_i = tl.load(L_block_ptr)
    Di = tl.load(D_block_ptr)

    dQ_acc = tl.zeros((Q_BLOCK, D), dtype=tl.float32)

    # Causal: Q block i never attends past its own diagonal K block.
    num_k_blocks = (q_block_id + 1) if is_causal else (Nk // K_BLOCK)

    for k_block_id in range(0, num_k_blocks):
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + batch_id * stride_kb,
            shape=(Nk, D), strides=(stride_kn, stride_kd),
            offsets=(k_block_id * K_BLOCK, 0),
            block_shape=(K_BLOCK, D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + batch_id * stride_vb,
            shape=(Nk, D), strides=(stride_vn, stride_vd),
            offsets=(k_block_id * K_BLOCK, 0),
            block_shape=(K_BLOCK, D), order=(1, 0),
        )
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        S_ij = tl.dot(Q_i, K_j.T) * scale

        if is_causal:
            q_idx = q_block_id * Q_BLOCK + tl.arange(0, Q_BLOCK)[:, None]
            k_idx = k_block_id * K_BLOCK + tl.arange(0, K_BLOCK)[None, :]
            causal_mask = q_idx >= k_idx
            P_ij = tl.where(causal_mask, tl.exp(S_ij - L_i), 0.0)
        else:
            P_ij = tl.exp(S_ij - L_i)

        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - Di)
        dS_cast = dS_ij.to(K_j.dtype)

        dQ_acc += tl.dot(dS_cast, K_j) * scale

    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + batch_id * stride_dqb,
        shape=(Nq, D), strides=(stride_dqn, stride_dqd),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D), order=(1, 0),
    )
    tl.store(dQ_block_ptr, dQ_acc.to(Q_i.dtype))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q, K, V: (B, N, D)
        """

        B, Nq, D = Q.shape
        Nk = K.shape[1]

        Q_BLOCK = 64
        K_BLOCK = 64

        # The block-pointer loads below don't do boundary checking, so a
        # partial last block would silently read/write out of bounds.
        assert Nq % Q_BLOCK == 0 and Nk % K_BLOCK == 0, (
            f"FlashAttentionTriton requires Nq ({Nq}) and Nk ({Nk}) to be "
            f"multiples of the block size ({Q_BLOCK})"
        )

        scale = D ** -0.5

        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device)

        grid = (Nq // Q_BLOCK, B)

        flash_attention_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=D,
            Q_BLOCK=Q_BLOCK,
            K_BLOCK=K_BLOCK,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.Q_BLOCK = Q_BLOCK
        ctx.K_BLOCK = K_BLOCK
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        Q_BLOCK = ctx.Q_BLOCK
        K_BLOCK = ctx.K_BLOCK

        B, Nq, D = Q.shape
        Nk = K.shape[1]

        dO = dO.contiguous()

        # D_i = rowsum(O_i * dO_i), the standard FlashAttention backward
        # correction term - cheap enough to do directly in PyTorch.
        Dsum = torch.sum(O * dO, dim=-1, keepdim=True).contiguous()  # (B, Nq, 1)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        grid_kv = (Nk // K_BLOCK, B)
        flash_attention_bwd_dkdv_kernel[grid_kv](
            Q, K, V, dO, L, Dsum,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            Dsum.stride(0), Dsum.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            Nq, Nk,
            scale,
            D=D, Q_BLOCK=Q_BLOCK, K_BLOCK=K_BLOCK, is_causal=is_causal,
        )

        grid_q = (Nq // Q_BLOCK, B)
        flash_attention_bwd_dq_kernel[grid_q](
            Q, K, V, dO, L, Dsum,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            Dsum.stride(0), Dsum.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            Nq, Nk,
            scale,
            D=D, Q_BLOCK=Q_BLOCK, K_BLOCK=K_BLOCK, is_causal=is_causal,
        )

        return dQ, dK, dV, None


if __name__ == "__main__":
    # ponytail: minimal runnable check; only meaningful on a CUDA box with
    # triton installed (this repo's dev machine has neither), so it's a
    # skip-if-unavailable smoke test rather than a hard assertion.
    if not torch.cuda.is_available():
        print("CUDA not available - skipping FlashAttentionTriton self-check.")
    else:
        torch.manual_seed(0)
        for is_causal in (False, True):
            B, N, D = 2, 128, 64
            device = "cuda"
            Q = torch.randn(B, N, D, device=device, dtype=torch.float32, requires_grad=True)
            K = torch.randn(B, N, D, device=device, dtype=torch.float32, requires_grad=True)
            V = torch.randn(B, N, D, device=device, dtype=torch.float32, requires_grad=True)

            Q2 = Q.detach().clone().requires_grad_()
            K2 = K.detach().clone().requires_grad_()
            V2 = V.detach().clone().requires_grad_()

            O = FlashAttentionTriton.apply(Q, K, V, is_causal)
            scale = D ** -0.5
            scores = (Q2 @ K2.transpose(-2, -1)) * scale
            if is_causal:
                mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
                scores = scores.masked_fill(~mask, float("-inf"))
            P = torch.softmax(scores, dim=-1)
            O_ref = P @ V2

            assert torch.allclose(O, O_ref, atol=1e-2, rtol=1e-2), "forward mismatch"

            O.sum().backward()
            O_ref.sum().backward()
            assert torch.allclose(Q.grad, Q2.grad, atol=1e-2, rtol=1e-2), "dQ mismatch"
            assert torch.allclose(K.grad, K2.grad, atol=1e-2, rtol=1e-2), "dK mismatch"
            assert torch.allclose(V.grad, V2.grad, atol=1e-2, rtol=1e-2), "dV mismatch"

        print("FlashAttentionTriton self-check passed.")
