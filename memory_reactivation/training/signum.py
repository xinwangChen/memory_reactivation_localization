"""Signum: EMA gradients followed by elementwise sign, with decoupled decay.

Update: m <- beta*m + (1-beta)*g;
        p <- (1-lr*weight_decay)*p - lr*sign(m).
Reference: https://proceedings.mlr.press/v80/bernstein18a.html
This is the single-EMA Signum variant, without Nesterov or shape scaling.
"""
from __future__ import annotations

from math import isfinite

import torch


class Signum(torch.optim.Optimizer):
    """Dense real parameters of all shapes use the same Signum update."""

    def __init__(self, params, lr: float, momentum: float = 0.9, weight_decay: float = 0.0):
        self._validate_options(lr, momentum, weight_decay)
        super().__init__(params, dict(lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay)))
        for group in self.param_groups:
            self._validate_options(group['lr'], group['momentum'], group['weight_decay'])
            for key in ('lr', 'momentum', 'weight_decay'):
                group[key] = float(group[key])
            if any(parameter.is_complex() for parameter in group['params']):
                raise ValueError('Signum requires real parameters')

    @staticmethod
    def _validate_options(lr, momentum, weight_decay):
        if not isfinite(float(lr)) or float(lr) < 0:
            raise ValueError('Signum learning rate must be finite and nonnegative')
        if not isfinite(float(momentum)) or not 0 <= float(momentum) < 1:
            raise ValueError('Signum momentum must be in [0, 1)')
        if not isfinite(float(weight_decay)) or float(weight_decay) < 0:
            raise ValueError('Signum weight_decay must be finite and nonnegative')

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Reject unsupported gradients before updating any parameter.
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None and (parameter.grad.is_sparse or parameter.grad.is_complex()):
                    raise RuntimeError('Signum requires dense real gradients')
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state['momentum_buffer'] = torch.zeros_like(parameter)
                momentum = state['momentum_buffer']
                momentum.lerp_(parameter.grad, 1.0 - group['momentum'])
                parameter.mul_(1.0 - group['lr'] * group['weight_decay'])
                parameter.add_(momentum.sign(), alpha=-group['lr'])
        return loss
