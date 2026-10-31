# mhc_pre_sinkhorn

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| cann | Unverified | Unsupported |
| torch_native | Unverified on NPU | Unverified on NPU |

## Signature

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

def mhc_pre_sinkhorn(x, phi, alpha, bias, hc_mult=4, num_iters=20, hc_eps=1e-6, norm_eps=1e-6, *, impl="auto"):
    ...
```

`auto` selects `cann`; no fallback.

## Description

Normalize flattened streams in FP32, project with phi, apply sigmoid to pre/post gates (post multiplier 2), then alternate row/column Sinkhorn normalization. No output RMSNorm is fused.

For u = flatten(x), z = (u / sqrt(mean(u**2) + norm_eps)) @ phi.T. Split z into pre/post/res, scale by alpha[0:3] and add the corresponding bias.

## Parameters and return value

| Parameter | Type | Constraint | Default |
| --- | --- | --- | --- |
| x | Tensor | [T,4,D], FP16/BF16 | required |
| phi | Tensor | [24,4*D], FP32 | required |
| alpha | Tensor | [3], FP32 | required |
| bias | Tensor | [24], FP32 | required |
| hc_mult | int | 4 only | 4 |
| num_iters | int | 20 only | 20 |
| hc_eps | float | finite, positive | 1e-6 |
| norm_eps | float | finite, positive | 1e-6 |
| impl | str | auto / cann / torch_native | auto |

All tensors must be contiguous, on one device (NPU for cann), ND layout; inputs are read-only. T=0 returns correctly typed empty tensors without launching CANN.

| Return | Type | Shape and dtype |
| --- | --- | --- |
| output | tuple[Tensor,Tensor,Tensor] | (hin [T,D] in x.dtype, hpost [T,4] FP32, hres [T,4,4] FP32) |

## cann

Requires CANN 9.1's `cann_ops_transformer.ops.mhc_pre_sinkhorn` public API. Imported and cached on first nonempty execution. Missing dependency and execution errors propagate; legacy custom_ops is never used. The registry restricts cann/auto to arch22 pending arch35 validation.

The facade expands [T,4,D] to [T,1,4,D] for the vendor BSND wrapper and squeezes the singleton sequence dimension of each output. D must be 128-aligned and <=100000.

## torch_native

Row softmax + hc_eps; divide by column sum + hc_eps; repeat (row sum + hc_eps, column sum + hc_eps) num_iters-1 times. hin = sum_i (sigmoid(pre_i)+hc_eps)*x_i.

FP32 intermediates, output activation cast to input dtype. An independent correctness baseline, not an automatic fallback. Accuracy thresholds and performance remain subject to NPU acceptance.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

x = torch.randn(1, 4, 4096, device="npu", dtype=torch.bfloat16)
phi = torch.randn(24, 16384, device="npu", dtype=torch.float32) / 128
alpha = torch.ones(3, device="npu", dtype=torch.float32)
bias = torch.zeros(24, device="npu", dtype=torch.float32)
y, post, comb = mhc_pre_sinkhorn(x, phi, alpha, bias, impl="cann")
```
