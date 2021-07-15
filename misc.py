import torch
from torch import nn

from copy import deepcopy
from collections import OrderedDict
from sys import stderr

# for type hint
from torch import Tensor

class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training:
            return self.model(inputs)
        else:
            return self.shadow(inputs)


class Warmup(object):
    """
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_params (list): warmup paramters
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.last_step = last_step

        # Find the period over which the warmup will run
        def warmup_period_fn(beta2):
            return int(1.0 / (1.0-beta2)) - 1
        self.warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        self.grads = [optimizer.param_groups[i]['lr']/self.warmup_period[i] for i in range(len(optimizer.param_groups))]

        # Initialise the low lr starting point
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.grads[i]

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.

        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self, step=None):
        """Dampen the learning rates.

        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        for i in range(len(self.optimizer.param_groups)):
            if step < self.warmup_period[i]:
                self.optimizer.param_groups[i]['lr'] += self.grads[i]