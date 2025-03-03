import torch
import triton
import triton.language as tl

def calc_compressed_len(x, stride, size):
    return  (x-size) // stride


def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': bm}, num_warps=nw) for bm in [16, 32, 64] for nw in [4, 8, 16]
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['num_heads', 'head_dim', 'block_stride', 'block_size'],
)
@triton.jit
def _compress_fwd(x, w, out, cu_input_len, cu_out_len, num_heads: tl.constexpr,
                  head_dim: tl.constexpr, block_stride: tl.constexpr, block_size: tl.constexpr,
                  BLOCK_M: tl.constexpr):
    bs_id, head_id, start_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_offset = tl.load(cu_input_len+bs_id)
    seq_upper = tl.load(cu_input_len+bs_id+1)
    out_offset = tl.load(cu_out_len+bs_id)
    out_upper = tl.load(cu_out_len+bs_id+1)
    out_len = out_upper - out_offset
    n_ctx = seq_upper-seq_offset
    
    x_ptr = x+seq_offset*num_heads*head_dim + head_id*head_dim
    out_ptr = out + out_offset*num_heads*head_dim + head_id*head_dim
    
    for task_id in range(start_id, (out_len+BLOCK_M-1)//BLOCK_M, tl.num_programs(2)):
        # task_x_offset = task_id*BLOCK_M
        off_m = tl.arange(0, BLOCK_M) + task_id*BLOCK_M
        off_n = tl.arange(0, head_dim)
        off_k = tl.arange(0, head_dim)
        task_out_ptr = out_ptr + off_m[:, None]*head_dim*num_heads + off_n[None, :]

        accumulator = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
        for inner_id in range(0, block_size, 1):
            off_acc_m = off_m * block_stride + inner_id
            task_w_ptr = w+inner_id*head_dim*head_dim

            x_data = tl.load(x_ptr+(off_acc_m*num_heads*head_dim)[:, None]+off_k[None,:], mask=off_acc_m[:,None]<n_ctx, other=0) # (BLOCK_M, head_dim)
            w_data = tl.load(task_w_ptr+off_k[:, None]*head_dim+off_n[None,:]) # head_dim, head_dim
            accumulator += tl.dot(x_data, w_data)
        accumulator = accumulator.to(tl.bfloat16)
        c_mask = (off_m[:, None]<out_len)
        tl.store(task_out_ptr, accumulator, mask=c_mask)



@triton.autotune(
    configs=get_autotune_config(),
    key=['num_heads', 'head_dim', 'block_stride', 'block_size'],
)
@triton.jit
def _compress_bwd_x(dx, w, out, cu_input_len, cu_out_len, num_heads: tl.constexpr,
                  head_dim: tl.constexpr, block_stride: tl.constexpr, block_size: tl.constexpr,
                  BLOCK_M: tl.constexpr):
    bs_id, head_id, start_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    out_offset = tl.load(cu_input_len+bs_id)
    out_upper = tl.load(cu_input_len+bs_id+1)
    seq_offset = tl.load(cu_out_len+bs_id)
    seq_upper = tl.load(cu_out_len+bs_id+1)
    out_len = out_upper - out_offset
    n_ctx = seq_upper-seq_offset
    
    x_ptr = dx+seq_offset*num_heads*head_dim + head_id*head_dim
    out_ptr = out + out_offset*num_heads*head_dim + head_id*head_dim
    
    for task_id in range(start_id, (out_len+BLOCK_M-1)//BLOCK_M, tl.num_programs(2)):
        # task_x_offset = task_id*BLOCK_M
        off_m = tl.arange(0, BLOCK_M) + task_id*BLOCK_M
        off_n = tl.arange(0, head_dim)
        off_k = tl.arange(0, head_dim)
        task_out_ptr = out_ptr + off_m[:, None]*head_dim*num_heads + off_n[None, :]

        accumulator = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
        for inner_id in range(0, block_size, 1):
            off_acc_m = off_m * block_stride + inner_id
            task_w_ptr = w+inner_id*head_dim*head_dim

            x_data = tl.load(x_ptr+(off_acc_m*num_heads*head_dim)[:, None]+off_k[None,:], mask=off_acc_m[:,None]<n_ctx, other=0) # (BLOCK_M, head_dim)
            w_data = tl.load(task_w_ptr+off_k[:, None]*head_dim+off_n[None,:]) # head_dim, head_dim
            accumulator += tl.dot(x_data, w_data)
        accumulator = accumulator.to(tl.bfloat16)
        c_mask = (off_m[:, None]<out_len)
        tl.store(task_out_ptr, accumulator, mask=c_mask)


# k/v: [num_token, NUM_HEAD, HEAD_DIM]
# w: [block_size*HEAD_DIM, HEAD_DIM]
class _compress_kv(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, k, v, w_k, w_v, cu_seq_len, block_stride, block_size
    ):
        NUM_HEAD, HEAD_DIM = k.shape[1:]
        cu_seq_len_cpu = cu_seq_len.tolist()
        pre = 0
        cu_out_len = [0]
        for x in cu_seq_len_cpu[1:]:
            cu_out_len.append(cu_out_len[-1] + calc_compressed_len(x-pre, block_stride, block_size))
            pre = x
        out_len = cu_out_len[-1]
        dtype = torch.bfloat16
        compressed_k = torch.empty(out_len, NUM_HEAD, HEAD_DIM, dtype=dtype, device=k.device)
        compressed_v = torch.empty(out_len, NUM_HEAD, HEAD_DIM, dtype=dtype, device=k.device)
        cu_out_len = torch.tensor(cu_out_len, device=cu_seq_len.device, dtype=torch.int32)
        
        grid = lambda args: (cu_seq_len.numel()-1, NUM_HEAD, 128)
        ctx.grid = grid
        _compress_fwd[grid](
            k, w_k, compressed_k, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size
        )
        _compress_fwd[grid](
            v, w_v, compressed_v, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size
        )

        ctx.save_for_backward(k, v, w_k, w_v, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size)
        return compressed_k, compressed_v
    
    
    
    @staticmethod
    def backward(ctx, dck, dcv):
        k, v, w_k, w_v, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size = ctx.saved_tensors
        NUM_HEAD, HEAD_DIM = k.shape[1:]
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        _compress_bwd_x[ctx.grid](
            dcv, w_v, dv, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size
        )
        _compress_bwd_x[ctx.grid](
            dck, w_k, dk, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size
        )

compress_kv = _compress_kv.apply