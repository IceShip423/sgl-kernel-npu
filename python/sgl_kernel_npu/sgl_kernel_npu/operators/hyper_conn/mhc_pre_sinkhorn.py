"""Inference mHC projection and Sinkhorn normalization."""

import math

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@register_impl(
    "hyper_conn.mhc_pre_sinkhorn", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "hyper_conn.mhc_pre_sinkhorn", "cann", supported_generations={Generation.ARCH22}
)
def _cann_impl(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps):
    global _cann_fn
    if x.device.type != "npu":
        raise ValueError("cann requires NPU tensors")
    if x.shape[-1] % 128 or x.shape[-1] > 100000:
        raise ValueError("cann requires hidden size <= 100000 and aligned to 128")
    if _cann_fn is None:
        from cann_ops_transformer.ops import mhc_pre_sinkhorn

        _cann_fn = mhc_pre_sinkhorn
    # CANN 9.1's arch22 wrapper requires BSND, even for packed tokens.
    y, post, comb = _cann_fn(
        x.unsqueeze(1), phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps
    )
    return y.squeeze(1).to(x.dtype), post.squeeze(1), comb.squeeze(1)


@register_impl("hyper_conn.mhc_pre_sinkhorn", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.mhc_pre_sinkhorn import (
            mhc_pre_sinkhorn,
        )

        _torch_native_fn = mhc_pre_sinkhorn
    return _torch_native_fn(*args)


@register_operator("hyper_conn.mhc_pre_sinkhorn")
def mhc_pre_sinkhorn(
    x,
    phi,
    alpha,
    bias,
    hc_mult=4,
    num_iters=20,
    hc_eps=1e-6,
    norm_eps=1e-6,
    *,
    impl="auto",
):
    """Project FP16/BF16 [T,4,D] streams; auto selects CANN 9.1 on arch22.

    phi is FP32 [24,4*D], alpha FP32 [3], bias FP32 [24]. Returns
    (hin [T,D] in x.dtype, hpost [T,4] FP32, hres [T,4,4] FP32).
    Inputs are contiguous on one device and remain unchanged. No output
    layer normalization is fused. Only hc_mult=4 and num_iters=20 are exposed.
    """
    if hc_mult != 4 or num_iters != 20:
        raise ValueError("mhc_pre_sinkhorn requires hc_mult=4 and num_iters=20")
    if x.ndim != 3 or x.shape[1] != hc_mult or x.shape[-1] == 0:
        raise ValueError("x must have shape [T,4,D] with D > 0")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must be float16 or bfloat16")
    for name, tensor, shape in (
        ("phi", phi, (24, hc_mult * x.shape[-1])),
        ("alpha", alpha, (3,)),
        ("bias", bias, (24,)),
    ):
        if tensor.shape != shape or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32 with shape {shape}")
    if any(
        t.device != x.device or not t.is_contiguous() for t in (x, phi, alpha, bias)
    ):
        raise ValueError("all inputs must be contiguous and on the same device")
    if any(not math.isfinite(eps) or eps <= 0 for eps in (hc_eps, norm_eps)):
        raise ValueError("hc_eps and norm_eps must be finite and positive")
    impl_fn = get_impl_by_name("hyper_conn.mhc_pre_sinkhorn", impl)
    if x.shape[0] == 0:
        return (
            x.new_empty((0, x.shape[-1])),
            x.new_empty((0, hc_mult), dtype=torch.float32),
            x.new_empty((0, hc_mult, hc_mult), dtype=torch.float32),
        )
    return impl_fn(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps)
