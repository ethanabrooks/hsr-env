# stdlib
from numbers import Number
import operator
from typing import Callable, Iterable, Union

# third party
import numpy as np

X = Union[Iterable, np.ndarray, Number, bool]
Key = Union[int, slice, np.ndarray]


def getitem(array_group, key: np.ndarray):
    if isinstance(array_group, np.ndarray):
        return array_group[key]
    return [getitem(a, key) for a in array_group]


def setitem(array_group: Union[list, np.ndarray], key: Key, x: X):
    if isinstance(array_group, np.ndarray):
        array_group[key] = x
    else:
        assert isinstance(x, Iterable)
        for _group, _x in zip(array_group, x):
            setitem(_group, key, _x)


def allocate(shapes: Iterable, pre_shape: tuple = None):
    pre_shape = pre_shape or ()
    try:
        return np.zeros(tuple(pre_shape) + shapes)
    except TypeError:
        return [allocate(shape, pre_shape) for shape in shapes]


def get_shapes(x, subset=None):
    if isinstance(x, np.ndarray):
        shape = np.shape(x)  # type: tuple
        if subset is None:
            return shape
        return shape[subset]
    if np.isscalar(x):
        return tuple()
    return [get_shapes(_x, subset) for _x in x]


def xnor(check: Callable, *vals: X):
    return all(map(check, vals)) or not any(map(check, vals))


def zip_op(op: Callable[[X, X], list],
           x: X,
           y: X,
           reduce_op: Callable[[Iterable[X]], X] = None):
    assert xnor(np.isscalar, [x, y])
    assert xnor(lambda z: isinstance(z, np.ndarray), [x, y])
    if isinstance(x, np.ndarray) or np.isscalar(x):
        z = op(x, y)
        if reduce_op:
            return reduce_op(z)
        return z
    assert len(x) == len(y)
    zipped = [zip_op(op, _x, _y, reduce_op) for _x, _y in zip(x, y)]
    if reduce_op:
        return reduce_op(zipped)
    return zipped


class ArrayGroup:
    @staticmethod
    def shape_like(x: X, pre_shape: tuple = None):
        return ArrayGroup(allocate(shapes=get_shapes(x), pre_shape=pre_shape))

    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key: Key):
        return ArrayGroup(getitem(self.values, key=key))

    def __setitem__(self, key: Key, value):
        if isinstance(value, ArrayGroup):
            value = value.values
        setitem(self.values, key=key, x=value)

    def __or__(self, other):
        return self.zip_op(operator.or_, other)

    def __eq__(self, other):
        return self.zip_op(operator.eq, other, np.all)

    def zip_op(self, op, other, reduce_op=None):
        assert callable(op)
        assert isinstance(other, ArrayGroup)
        return ArrayGroup(zip_op(op, self.values, other.values, reduce_op))

    @property
    def shape(self):
        return get_shapes(self.values)
