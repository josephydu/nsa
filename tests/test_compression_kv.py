import torch
from nsa.compression_kv import compress_kv

conv1d = torch.nn.functional.conv1d


bs, seqlen, head_dim, kv_num_head = 1, 1024, 128, 2
block_size, block_stride = 64, 16
dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.manual_seed(3)

k = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype, requires_grad=True)
v = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype, requires_grad=True)
w_k = torch.randn(block_size*head_dim, head_dim, dtype=dtype, requires_grad=True)
w_v = torch.randn(block_size*head_dim, head_dim, dtype=dtype, requires_grad=True)
seq_len = torch.Tensor([0] + [seqlen] * bs)
cu_seq_len = torch.cumsum(seq_len, dim=0).to(torch.int32).to(device)

c_k, c_v =  compress_kv(k, v, w_k, w_v, cu_seq_len, block_stride, block_size)




def compute_reference_kv(k, w_k, cu_seq_len, block_size, block_stride):
    """Torch实现的参考计算逻辑"""
    ref_k = torch.zeros_like(c_k)
    out_idx = 0
    bs = len(cu_seq_len) - 1
    
    # 保存中间结果用于反向计算
    saved_for_backward = []
    
    for i in range(bs):
        start_idx = int(cu_seq_len[i])
        end_idx = int(cu_seq_len[i+1])
        seq_len = end_idx - start_idx
        single_k = k[start_idx:end_idx, :, :]
        
        for w in range((seq_len - block_size) // block_stride):
            w_start = w * block_stride
            w_end = w_start + block_size
            k_window = single_k[w_start:w_end, :, :]
            
            for h in range(kv_num_head):
                single_head_k = k_window[:, h, :]
                single_head_k = single_head_k.reshape(1, -1)
                # 保存输入矩阵和权重用于反向
                saved_for_backward.append((single_head_k, w_k))
                ref_k[out_idx, h, :] = torch.matmul(single_head_k, w_k)
                
            out_idx += 1
            
    # 为反向计算保存必要信息
    ref_k.saved_for_backward = saved_for_backward
    return ref_k

def compute_reference_dw_k(ref_k):
    """计算参考的权重梯度"""
    dw_k = torch.zeros_like(w_k)
    # 对每个保存的窗口计算梯度
    for single_head_k, w in ref_k.saved_for_backward:
        # 构造可求导变量
        k_tensor = single_head_k.detach().requires_grad_(True)
        w_tensor = w.detach().requires_grad_(True)
        # 前向计算
        output = torch.matmul(k_tensor, w_tensor)
        # 反向传播（假设梯度为1）
        output.backward(torch.ones_like(output))
        # 累加梯度
        dw_k += w_tensor.grad
    return dw_k

# 前向测试
ref_k = compute_reference_kv(k, w_k, cu_seq_len, block_size, block_stride)
torch.testing.assert_close(c_k, ref_k, rtol=1e-2, atol=1e-2)
print("Forward Passed")

# 反向测试
# 计算参考梯度
reference_dw_k = compute_reference_dw_k(ref_k)


grad_output = torch.ones_like(c_k)

c_k.backward(gradient=grad_output)


# 比较梯度
print("Computed grad sum:", w_k.grad.sum().item())
print("Reference grad sum:", reference_dw_k.sum().item())
print("Computed grad mean:", w_k.grad.mean().item())
print("Reference grad mean:", reference_dw_k.mean().item())
torch.testing.assert_close(w_k.grad, reference_dw_k.to(w_k.dtype), rtol=1e-2, atol=1e-2)
print("Backward Passed")

