# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

from typing import Optional

import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.components.activations import Activation
from xformers.triton.activations import (
    get_triton_activation_bwd_kernel,
    get_triton_activation_kernel,
)
from xformers.triton.k_dropout import k_dropout_bw, k_dropout_fw


# Helper to handle the SPMD launch grid and error cases
class _dropout(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, p, bias, activation, activation_grad, memory_efficient):
        # Soft-flatten an hypothetical 3rd dimension
        x_ = x.reshape(-1, x.shape[-1]).contiguous()
        y = torch.empty_like(x_)
        _, N = x_.shape

        assert bias is None or bias.dtype == x.dtype, bias

        # Generate one seed per sample
        # seed max is int32 max for positive numbers: 2**16
        seeds = (
            torch.randint(65536, (x_.shape[0],), device=x.device).to(torch.int32)
            if memory_efficient
            else x_
        )
        mask = torch.empty_like(x_).to(torch.bool) if not memory_efficient else x_

        # SPMD launch grid
        def grid(meta):
            return (
                x_.shape[0],
                triton.cdiv(x_.shape[1], meta["BLOCK_SIZE"]),
            )

        # fmt: off
        k_dropout_fw[grid](
            y, mask, x_, bias if bias is not None else x_,
            seeds,
            y.stride(0),
            N,
            p,
            USE_BIAS=bias is not None,
            ACTIVATION=activation,
            SAVE_MASK=not memory_efficient
        )
        # fmt: on

        ctx.save_for_backward(
            seeds if memory_efficient else None,
            bias,
            x if activation is not None else None,
            mask if not memory_efficient else None,
        )

        ctx.trainable_bias = bias is not None
        ctx.activation_grad = activation_grad
        ctx.memory_efficient = memory_efficient
        ctx.p = p

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        (seeds, bias, inputs, mask) = ctx.saved_tensors

        # Soft-flatten an hypothetical 3rd dimension
        grad_out_ = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_in = torch.empty_like(grad_out_)

        _, N = grad_out_.shape

        # Optional inputs to compute the activation contribution to the gradient
        assert inputs is not None or ctx.activation_grad is None

        if inputs is None:
            inputs = grad_out_
        elif inputs.ndim > 2:
            inputs = inputs.reshape(-1, grad_out.shape[-1])

        # SPMD launch grid
        def grid(meta):
            return (
                grad_out_.shape[0],
                triton.cdiv(grad_out_.shape[1], meta["BLOCK_SIZE"]),
            )

        # fmt: off
        k_dropout_bw[grid](
            grad_in, grad_out_, inputs,
            mask if mask is not None else inputs,
            bias if bias is not None else inputs,
            seeds if seeds is not None else inputs,
            grad_out_.stride(0), inputs.stride(0),
            N,
            ctx.p,
            USE_BIAS=bias is not None,
            ACTIVATION_GRAD=ctx.activation_grad,
            SAVED_MASK=mask is not None)
        # fmt: on

        if ctx.trainable_bias:
            grad_bias: Optional[torch.Tensor] = torch.sum(grad_in, dim=0)
        else:
            grad_bias = None

        return grad_in.reshape_as(grad_out), None, grad_bias, None, None, None


def dropout(
    x: torch.Tensor,
    p: float,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[Activation] = None,
    memory_efficient: bool = False,
):
    """
    Apply dropout on the input tensor.
    Optionally add a bias, the computation will be fused.

    .. note: Memory efficient trades off speed for memory use during training,
        it has no effect on inference speed.

    """

    # Micro optim, skip dropout
    if p == 0.0 and activation is None:
        return x + bias if bias is not None else x

    act_kernel = get_triton_activation_kernel(activation)
    act_grad_kernel = get_triton_activation_bwd_kernel(activation)

    if not x.requires_grad:
        # We're not training, make sure that the inputs are not saved
        memory_efficient = True

    return _dropout.apply(x, p, bias, act_kernel, act_grad_kernel, memory_efficient)


class FusedDropoutBias(torch.nn.Module):
    def __init__(
        self,
        p: float,
        bias_shape: Optional[int],
        activation: Optional[Activation] = None,
        memory_efficient: bool = False,  # default to speed
    ) -> None:
        """
        A Fused dropout + activation + bias layer.
        The operationg ordering is `y = dropout(activation(x+bias))`

        .. note: Memory efficient trades off speed for memory use during training,
            it has no effect on inference speed.

        """
        super().__init__()
        self.p = p
        self.activation = activation
        self.register_buffer(
            "bias", torch.zeros(bias_shape) if bias_shape is not None else None
        )
        self.activation = get_triton_activation_kernel(activation)
        self.activation_grad = get_triton_activation_bwd_kernel(activation)
        self.memory_efficient = memory_efficient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p if self.training else 0.0

        # Make sure that the activations are not saved if we're not training anyway
        memory_efficient = (
            self.memory_efficient if (self.training and x.requires_grad) else True
        )

        return _dropout.apply(
            x, p, self.bias, self.activation, self.activation_grad, memory_efficient
        )
