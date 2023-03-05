__author__ = "Jijia Wu"
__credits__ = ["Jijia Wu"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jijia Wu"
__email__ = "jijiawu.cs@gmail.com"


import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class SAdam(Optimizer):
    r"""Implements SAdam (Scalable Adam) algorithm.

    .. explain::
        It has been proven that the Adam optimizer can converge to
        the minimum in deep learning neural networks, however, its
        stride of each step can sometimes be too small. This means that
        more steps are required to converge to the desired result.

        SAdam (Scalable Adam) is an experimental optimizer that we
        mathematically replace each parameters (p) in the model with
            ``` p = a * exp(b) ```
        which means the convergence of Adam still holds since
        we're implicitly optimizing the `a` and `b`


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        capturable (bool, optional): whether this instance is safe to capture in a CUDA graph.
            Passing True can impair ungraphed performance, so if you don't intend to
            graph capture this instance, leave it False (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        if capturable:
            raise ValueError(
                "SAdam(capturable=True) is not supported currently, we will fix it soon")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable)
        super(SAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            ln_scales = []
            ln_scale_exp_avgs = []
            ln_scale_exp_avg_sqs = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'SAdam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                            if self.defaults['capturable'] else torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p)

                        state['ln_scale'] = torch.zeros_like(p)
                        state['ln_scale_exp_avg'] = torch.zeros_like(p)
                        state['ln_scale_exp_avg_sq'] = torch.zeros_like(p)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    ln_scales.append(state['ln_scale'])
                    ln_scale_exp_avgs.append(state['ln_scale_exp_avg'])
                    ln_scale_exp_avg_sqs.append(state['ln_scale_exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state_steps.append(state['step'])

            sadam(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  state_steps,
                  amsgrad=group['amsgrad'],
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'],
                  capturable=group['capturable'],

                  ln_scales=ln_scales,
                  ln_scale_exp_avgs=ln_scale_exp_avgs,
                  ln_scale_exp_avg_sqs=ln_scale_exp_avg_sqs,
                  )

        return loss


def sadam(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[Tensor],
          # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
          # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
          foreach: bool = None,
          capturable: bool = False,
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool,


          ln_scales,
          ln_scale_exp_avgs,
          ln_scale_exp_avg_sqs,
          ):
    r"""Functional API that performs SAdam algorithm computation.
    See :class:`~sadam.SAdam` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError(
            'torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        raise ValueError("_multi_tensor_sadam not implemented yet")
        # func = _multi_tensor_sadam
    else:
        func = _single_tensor_sadam

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable,
         ln_scales=ln_scales,
         ln_scale_exp_avgs=ln_scale_exp_avgs,
         ln_scale_exp_avg_sqs=ln_scale_exp_avg_sqs)


def _single_tensor_sadam(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         max_exp_avg_sqs: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         amsgrad: bool,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float,
                         eps: float,
                         maximize: bool,
                         capturable: bool,
                         ln_scales,
                         ln_scale_exp_avgs,
                         ln_scale_exp_avg_sqs,):

    for i, param in enumerate(params):
        ln_scale = ln_scales[i]
        ln_scale_exp_avg = ln_scale_exp_avgs[i]
        ln_scale_exp_avg_sq = ln_scale_exp_avg_sqs[i]

        scale = torch.exp(ln_scale)
        par = param / scale

        g = grads[i] if not maximize else -grads[i]
        p_grad = g * scale
        s_grad = g * param  # g * par * scale

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors."

        # update step
        step_t += 1

        if weight_decay != 0:
            p_grad = p_grad.add(param * scale, alpha=weight_decay)
            s_grad = s_grad.add(param * param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(p_grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, p_grad, p_grad)

        ln_scale_exp_avg.mul_(beta1).add_(s_grad, alpha=1 - beta1)
        ln_scale_exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, s_grad, s_grad)

        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = math.sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sqs[i],
                      exp_avg_sq, out=max_exp_avg_sqs[i])

            # Use the max. for normalizing running avg. of gradient
            p_denom = (max_exp_avg_sqs[i].sqrt() /
                       bias_correction2_sqrt).add_(eps)
            s_denom = (ln_scale_exp_avg_sq.sqrt() /
                       bias_correction2_sqrt).add_(eps)
        else:
            p_denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            s_denom = (ln_scale_exp_avg_sq.sqrt() /
                       bias_correction2_sqrt).add_(eps)

        new_par = par - step_size * (exp_avg / p_denom)
        new_scale = ln_scale - \
            torch.clamp(step_size * (ln_scale_exp_avg / s_denom), -0.5, 0.5)
        torch.clamp(new_scale, -10, 10, out=ln_scale)
        torch.mul(new_par, torch.exp(ln_scale), out=param)
