import torch
from nsa.compression_kv import compress_kv, calc_compressed_len
from typing import Tuple

# 定义卷积操作
conv1d = torch.nn.functional.conv1d

# 初始化测试参数
BATCH_SIZE = 5
SEQ_LENGTH = 1024 * 32
HEAD_DIM = 128
KV_NUM_HEADS = 2
BLOCK_SIZE = 64
BLOCK_STRIDE = 16
DTYPE = torch.bfloat16
DEVICE = "cuda"

def initialize_test_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """初始化测试数据"""
    torch.set_default_device(DEVICE)
    torch.manual_seed(3)

    # 初始化输入张量
    k = torch.randn(BATCH_SIZE * SEQ_LENGTH, KV_NUM_HEADS, HEAD_DIM, dtype=DTYPE, requires_grad=True)
    v = torch.randn(BATCH_SIZE * SEQ_LENGTH, KV_NUM_HEADS, HEAD_DIM, dtype=DTYPE, requires_grad=True)
    w_k = torch.randn(BLOCK_SIZE * HEAD_DIM, HEAD_DIM, dtype=DTYPE, requires_grad=True)
    w_v = torch.randn(BLOCK_SIZE * HEAD_DIM, HEAD_DIM, dtype=DTYPE, requires_grad=True)
    
    # 初始化序列长度信息
    seq_len = torch.Tensor([0] + [SEQ_LENGTH] * BATCH_SIZE)
    cu_seq_len = torch.cumsum(seq_len, dim=0).to(torch.int32).to(DEVICE)
    
    return k, v, w_k, w_v, cu_seq_len

def compute_reference_kv(input_tensor: torch.Tensor, 
                        weight: torch.Tensor,
                        cu_seq_len: torch.Tensor,
                        block_size: int,
                        block_stride: int) -> torch.Tensor:
    """参考实现的计算逻辑，支持自动求导"""
    num_heads = input_tensor.size(1)
    result_list = []
    batch_size = len(cu_seq_len) - 1
    
    for i in range(batch_size):
        start_idx = int(cu_seq_len[i])
        end_idx = int(cu_seq_len[i+1])
        seq_len = end_idx - start_idx
        single_seq = input_tensor[start_idx:end_idx, :, :]
        
        num_windows = calc_compressed_len(seq_len, block_stride, block_size)
        for w in range(num_windows):
            w_start = w * block_stride
            w_end = w_start + block_size
            window = single_seq[w_start:w_end, :, :]
            
            head_results = []
            for h in range(num_heads):
                single_head = window[:, h, :]
                single_head_flat = single_head.reshape(1, -1)
                head_result = torch.matmul(single_head_flat, weight)
                head_results.append(head_result)
            
            window_result = torch.stack(head_results, dim=1)
            result_list.append(window_result)
    
    return torch.cat(result_list, dim=0)

def test_forward_pass(k: torch.Tensor, v: torch.Tensor, 
                     w_k: torch.Tensor, w_v: torch.Tensor,
                     cu_seq_len: torch.Tensor) -> None:
    """测试前向传播"""
    c_k, c_v = compress_kv(k, v, w_k, w_v, cu_seq_len, BLOCK_STRIDE, BLOCK_SIZE)
    
    ref_k = compute_reference_kv(k, w_k, cu_seq_len, BLOCK_SIZE, BLOCK_STRIDE)
    ref_v = compute_reference_kv(v, w_v, cu_seq_len, BLOCK_SIZE, BLOCK_STRIDE)
    
    torch.testing.assert_close(c_k, ref_k, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(c_v, ref_v, rtol=1e-2, atol=1e-2)
    print("Forward Passed")

def test_backward_pass(k: torch.Tensor, v: torch.Tensor,
                      w_k: torch.Tensor, w_v: torch.Tensor,
                      cu_seq_len: torch.Tensor) -> None:
    """测试反向传播"""
    # 测试k的反向传播
    target_k = torch.randn_like(compress_kv(k, v, w_k, w_v, cu_seq_len, BLOCK_STRIDE, BLOCK_SIZE)[0])
    
    # 参考实现
    ref_k = compute_reference_kv(k, w_k, cu_seq_len, BLOCK_SIZE, BLOCK_STRIDE)
    ref_loss_k = torch.mean((ref_k - target_k) ** 2)
    ref_loss_k.backward()
    ref_wk_grad = w_k.grad.clone()
    
    # 清空梯度并重新计算
    for param in [w_k, w_v, k, v]:
        param.grad = None
        
    c_k, _ = compress_kv(k.detach().requires_grad_(True), 
                        v.detach().requires_grad_(True), 
                        w_k, w_v, cu_seq_len, BLOCK_STRIDE, BLOCK_SIZE)
    
    c_loss_k = torch.mean((c_k - target_k) ** 2)
    c_loss_k.backward(retain_graph=True)
    c_wk_grad = w_k.grad.clone()
    
    torch.testing.assert_close(c_wk_grad, ref_wk_grad, rtol=2e-2, atol=2e-2)
    print("Backward for K Passed")

    # 测试v的反向传播
    target_v = torch.randn_like(compress_kv(k, v, w_k, w_v, cu_seq_len, BLOCK_STRIDE, BLOCK_SIZE)[1])
    
    # 清空梯度并重新计算
    for param in [w_k, w_v, k, v]:
        param.grad = None
        
    _, c_v = compress_kv(k.detach().requires_grad_(True), 
                        v.detach().requires_grad_(True), 
                        w_k, w_v, cu_seq_len, BLOCK_STRIDE, BLOCK_SIZE)
    
    c_loss_v = torch.mean((c_v - target_v) ** 2)
    c_loss_v.backward()
    c_wv_grad = w_v.grad.clone()
    
    # 参考实现
    for param in [w_k, w_v, k, v]:
        param.grad = None
        
    ref_v = compute_reference_kv(v, w_v, cu_seq_len, BLOCK_SIZE, BLOCK_STRIDE)
    ref_loss_v = torch.mean((ref_v - target_v) ** 2)
    ref_loss_v.backward()
    ref_wv_grad = w_v.grad.clone()
    
    torch.testing.assert_close(c_wv_grad, ref_wv_grad, rtol=2e-2, atol=2e-2)
    print("Backward for V Passed")

def main():
    """主测试函数"""
    k, v, w_k, w_v, cu_seq_len = initialize_test_data()
    
    # 测试前向传播
    test_forward_pass(k, v, w_k, w_v, cu_seq_len)
    
    # 测试反向传播
    test_backward_pass(k, v, w_k, w_v, cu_seq_len)

if __name__ == "__main__":
    main()
