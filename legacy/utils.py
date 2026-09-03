import operator
from collections import OrderedDict
from numbers import Number
import torch
import copy
import pdb
from typing import Dict, List, Union
import numpy as np
import warnings

from wilds.common.grouper import Grouper
from wilds.common.utils import get_counts
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset
from collections import defaultdict
import random

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def sign(self):
        return ParamDict({k: torch.sign(v) for k, v in self.items()})
    
    def ge(self, number):
        return ParamDict({k: torch.ge(v, number) for k, v in self.items()})
    
    def le(self, number):
        return ParamDict({k: torch.le(v, number) for k, v in self.items()})
    
    def gt(self, number):
        return ParamDict({k: torch.gt(v, number) for k, v in self.items()})
    
    def lt(self, number):
        return ParamDict({k: torch.lt(v, number) for k, v in self.items()})
    
    def abs(self):
         return ParamDict({k: torch.abs(v) for k, v in self.items()})

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)

    def to(self, device):
        return ParamDict({k: v.to(device) for k, v in self.items()})
