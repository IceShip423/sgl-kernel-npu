"""Inference mHC post mapping, with the CANN public argument order."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@register_impl("hyper_conn.mhc_post", "auto", supported_generations={Generation.ARCH22})
@register_impl("hyper_conn.mhc_post", "cann", supported_generations={Generation.ARCH22})
def _cann_impl(x, h_res, h_out, h_post):
    global _cann_fn
    if x.device.type != "npu":
        raise ValueError("cann requires NPU tensors")
    if _cann_fn is None:
        from cann_ops_transformer.ops import mhc_post

        _cann_fn = mhc_post
    return _cann_fn(x, h_res, h_out, h_post)


@register_impl("hyper_conn.mhc_post", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.mhc_post import mhc_post

        _torch_native_fn = mhc_post
    return _torch_native_fn(*args)


@register_operator("hyper_conn.mhc_post")
def mhc_post(x, h_res, h_out, h_post, *, impl="auto"):
    """Return [T,4,D] post-mapped streams; auto selects CANN 9.1 on arch22.

    x [T,4,D] and h_out [T,D] are FP16/BF16; h_res [T,4,4] and
    h_post [T,4] are FP32. All inputs are contiguous on one device.
    Result uses x.dtype; inputs remain unchanged.
    y[t,i,d] = h_post[t,i]*h_out[t,d] + sum_j h_res[t,i,j]*x[t,j,d].
    """
    if x.ndim != 3 or x.shape[1] != 4 or x.shape[-1] == 0:
        raise ValueError("x must have shape [T,4,D] with D > 0")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must be float16 or bfloat16")
    t, n, d = x.shape
    for name, tensor, shape, dtype in (
        ("h_res", h_res, (t, n, n), torch.float32),
        ("h_out", h_out, (t, d), x.dtype),
        ("h_post", h_post, (t, n), torch.float32),
    ):
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if any(
        t.device != x.device or not t.is_contiguous() for t in (x, h_res, h_out, h_post)
    ):
        raise ValueError("all inputs must be contiguous and on the same device")
    impl_fn = get_impl_by_name("hyper_conn.mhc_post", impl)
    if x.shape[0] == 0:
        return torch.empty_like(x)
    return impl_fn(x, h_res, h_out, h_post)
