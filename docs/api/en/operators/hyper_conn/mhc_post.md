# mhc_post

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| cann | Unverified | Unsupported |
| torch_native | Unverified on NPU | Unverified on NPU |

## Signature

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_post

def mhc_post(x, h_res, h_out, h_post, *, impl="auto"):
    ...
```

`auto` selects `cann`; no fallback.

## Description

Compute y[t,i,d] = h_post[t,i] * h_out[t,d] + sum_j h_res[t,i,j] * x[t,j,d] in FP32, then cast once to x.dtype.

## Parameters and return value

| Parameter | Type | Constraint | Default |
| --- | --- | --- | --- |
| x | Tensor | [T,4,D], FP16/BF16 | required |
| h_res | Tensor | [T,4,4], FP32 | required |
| h_out | Tensor | [T,D], same dtype as x | required |
| h_post | Tensor | [T,4], FP32 | required |
| impl | str | auto / cann / torch_native | auto |

All tensors must be contiguous, on one device (NPU for cann), ND layout; inputs are read-only. T=0 returns correctly typed empty tensors without launching CANN.

| Return | Type | Shape and dtype |
| --- | --- | --- |
| output | Tensor | y [T,4,D], same dtype as x |

## cann

Requires CANN 9.1's `cann_ops_transformer.ops.mhc_post` public API. Imported and cached on first nonempty execution. Missing dependency and execution errors propagate; legacy custom_ops is never used. The registry restricts cann/auto to arch22 pending arch35 validation.

Argument order matches the vendor API: residual streams, residual matrix, layer output, post gates.

## torch_native

Independent torch.bmm residual mapping plus post gating; no vendor calls.

FP32 intermediates, output activation cast to input dtype. An independent correctness baseline, not an automatic fallback. Accuracy thresholds and performance remain subject to NPU acceptance.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_post

x = torch.randn(1, 4, 4096, device="npu", dtype=torch.bfloat16)
h_res = torch.eye(4, device="npu", dtype=torch.float32).unsqueeze(0)
h_out = torch.zeros(1, 4096, device="npu", dtype=x.dtype)
h_post = torch.ones(1, 4, device="npu", dtype=torch.float32)
y = mhc_post(x, h_res, h_out, h_post, impl="cann")
```
