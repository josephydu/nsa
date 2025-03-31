import torch
import math
import triton
import triton.language as tl



# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True





@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)



# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(q.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq



def test_dq_kernel_directly():
    B, T, H, D = 1, 32768, 64, 128
    torch.manual_seed(42)
    
    q = torch.randn(B, T, H, D, device='cuda')
    K = torch.randn(B, T, H, D, device='cuda')
    V = torch.randn(B, T, H, D, device='cuda')
    do = torch.randn(B, T, H, D, device='cuda')
    
    m = torch.randn(B, H, T, device='cuda') 
    D = torch.randn(B, H, T, device='cuda') 
    
    dq = torch.zeros_like(q)
    
    HEAD_DIM = D
    BLOCK_M2, BLOCK_N2 = 128, 32
    num_steps = T // BLOCK_N2
    
    HEAD_DIM = 128  # Should be fixed value from D dimension
    BLOCK_M2, BLOCK_N2 = 128, 32
    num_steps = T // BLOCK_N2
    
    grid = (triton.cdiv(T, BLOCK_M2), B*H)
    _attn_bwd_dq[grid](
        dq, q, K, V,
        do, m, D,  # D here is the delta tensor
        stride_tok=H*D,  
        stride_d=1,    
        H=H,
        N_CTX=T,
        BLOCK_M2=BLOCK_M2,
        BLOCK_N2=BLOCK_N2,
        HEAD_DIM=HEAD_DIM,  # Use the fixed value
        start_m=0,
        start_n=0,
        num_steps=num_steps,
        MASK=False
    )
    
    # 验证结果
    print("\nDirect DQ Kernel Test:")
    print(f"Output shape: {dq.shape}")  # [1,32768,64,128]
    print(f"Max value: {dq.max().item():.4f}")
    print(f"Min value: {dq.min().item():.4f}")
    print(f"Mean absolute: {torch.abs(dq).mean().item():.6f}")
    
    # 检查内存布局
    assert dq.stride() == (32768*64*128, 64*128, 128, 1), "Stride mismatch!"
    
    return dq

if __name__ == "__main__":
    dq = test_dq_kernel_directly()
    print(dq)