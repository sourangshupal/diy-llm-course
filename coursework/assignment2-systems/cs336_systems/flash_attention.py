import torch

class FlashAttentionAutograd(torch.autograd.Function):
    """
    FlashAttention (forward + backward)

    Inputs:
        Q: (B, Nq, d)
        K: (B, Nk, d)
        V: (B, Nk, d)

    Output:
        O: (B, Nq, d)
    """

    @staticmethod
    def forward(ctx, Q, K, V):
        B, Nq, d = Q.shape
        Nk = K.shape[1]

        # -----------------------------
        # block size
        # -----------------------------
        Bq = 64  # Query block size
        Bk = 64  # Key / Value block size

        # Split the KV matrices into small blocks

        Tq = -(-Nq // Bq)  # number of Q blocks (ceil div; last block may be partial)
        Tk = -(-Nk // Bk)  # number of K/V blocks (ceil div; last block may be partial)

        scale = d ** -0.5

        # -----------------------------
        # outputs and intermediate quantities
        # -----------------------------
        O = torch.zeros_like(Q)              # attention output
        L = torch.zeros(B, Nq, device=Q.device)  # log-sum-exp (for backward)

        # =========================================================
        # explicit loop over the batch dimension
        # =========================================================
        for b in range(B):
            Q_b = Q[b]  # (Nq, d)
            K_b = K[b]  # (Nk, d)
            V_b = V[b]  # (Nk, d)

            # =====================================================
            # outer loop: Query block
            # =====================================================
            for i in range(Tq):
                q_i = Q_b[i * Bq:(i + 1) * Bq]  # (bq, d) - bq == Bq except possibly the last block
                bq = q_i.shape[0]

                # -----------------------------
                # FlashAttention core state
                # -----------------------------
                m = torch.full((bq, 1), -float("inf"), device=Q.device)  # running max
                l = torch.zeros((bq, 1), device=Q.device)               # running sum
                O_acc = torch.zeros((bq, d), device=Q.device)           # unnormalized output accumulation

                # =================================================
                # inner loop: Key / Value block
                # =================================================
                for j in range(Tk):
                    k_j = K_b[j * Bk:(j + 1) * Bk]  # (Bk, d)
                    v_j = V_b[j * Bk:(j + 1) * Bk]  # (Bk, d)

                    # --------------------------------------------
                    # compute local attention score
                    # S_ij = Q_i @ K_j^T / sqrt(d)
                    # --------------------------------------------
                    S = q_i @ k_j.T * scale  # (Bq, Bk)

                    # max per row inside the current block
                    row_max = S.max(dim=1, keepdim=True).values  # (Bq, 1)

                    # update running max
                    m_new = torch.maximum(m, row_max)

                    # --------------------------------------------
                    # numerically stable softmax (key to FlashAttention)
                    # --------------------------------------------
                    # correct contribution from the old block
                    exp_old = torch.exp(m - m_new) * l

                    # exp for the current block
                    P = torch.exp(S - m_new)

                    # update denominator
                    l = exp_old + P.sum(dim=1, keepdim=True)

                    # update unnormalized output
                    O_acc = (
                        torch.exp(m - m_new) * O_acc +
                        P @ v_j
                    )

                    # write back running max
                    m = m_new

                # =================================================
                # softmax normalization within the block
                # =================================================
                O_i = O_acc / l  # (Bq, d)

                # log-sum-exp: L = m + log(l)
                L_i = m.squeeze(1) + torch.log(l.squeeze(1))

                # write back to global output
                O[b, i * Bq:(i + 1) * Bq] = O_i
                L[b, i * Bq:(i + 1) * Bq] = L_i

        # save variables needed for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.Bq = Bq
        ctx.Bk = Bk

        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale
        Bq = ctx.Bq
        Bk = ctx.Bk

        B, Nq, d = Q.shape
        Nk = K.shape[1]

        Tq = -(-Nq // Bq)
        Tk = -(-Nk // Bk)

        # --------------------------------------------
        # important intermediate quantity in FlashAttention backward
        # D_i = sum_j O_ij * dO_ij
        # --------------------------------------------
        D = torch.sum(O * dO, dim=-1)  # (B, Nq)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for b in range(B):
            for i in range(Tq):
                q_i = Q[b, i * Bq:(i + 1) * Bq]
                dO_i = dO[b, i * Bq:(i + 1) * Bq]
                L_i = L[b, i * Bq:(i + 1) * Bq]
                D_i = D[b, i * Bq:(i + 1) * Bq]

                for j in range(Tk):
                    k_j = K[b, j * Bk:(j + 1) * Bk]
                    v_j = V[b, j * Bk:(j + 1) * Bk]

                    # --------------------------------------------
                    # recompute local attention probability P_ij
                    # P_ij = exp(S_ij - L_i)
                    # --------------------------------------------
                    S = q_i @ k_j.T * scale
                    P = torch.exp(S - L_i.unsqueeze(1))

                    # -------- dV --------
                    dV[b, j * Bk:(j + 1) * Bk] += P.T @ dO_i

                    # -------- dS --------
                    dP = dO_i @ v_j.T
                    dS = P * (dP - D_i.unsqueeze(1))

                    # -------- dQ --------
                    dQ[b, i * Bq:(i + 1) * Bq] += dS @ k_j * scale

                    # -------- dK --------
                    dK[b, j * Bk:(j + 1) * Bk] += dS.T @ q_i * scale

        return dQ, dK, dV
