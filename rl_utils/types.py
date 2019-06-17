# stdlib
from collections import namedtuple
from typing import Any, Sequence, Union

# third party
import numpy as np

Shape = Union[int, Sequence[int]]

Obs = Any


class Step(namedtuple('Step', 'o1 a r o2 t')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


ArrayLike = Union[np.ndarray, list]
