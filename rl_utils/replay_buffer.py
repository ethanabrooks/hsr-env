# stdlib
from typing import Iterable

# third party
import numpy as np

# first party
from rl_utils.array_group import ArrayGroup, Key, X


def get_index(value):
    if np.isscalar(value):
        return 1
    if isinstance(value, np.ndarray):
        try:
            return value.shape[0]
        except IndexError:
            return 1
    assert isinstance(value, Iterable)
    indices = set(map(get_index, value))
    if len(indices) == 1:  # all the same
        return indices.pop()
    else:
        return 1


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = None
        self.full = False
        self.pos = 0

    @property
    def empty(self):
        return self.buffer is None

    def __getitem__(self, key: Key):
        assert self.buffer is not None
        return self.buffer[self.modulate(key)]

    def __setitem__(self, key: Key, value):
        self.buffer[self.modulate(key)] = value

    def __len__(self):
        return self.maxlen if self.full else self.pos

    def modulate(self, key: Key):
        if isinstance(key, slice):
            key = np.arange(key.start or 0,
                            0 if key.stop is None else key.stop, key.step)
        return (key + self.pos) % self.maxlen

    def array(self):
        if self.buffer is None:
            return np.empty(0)
        return self[-len(self): 0].values

    def sample(self, batch_size: int, seq_len: int = None):
        # indices are negative because indices are relative to pos
        indices = np.random.randint(
            -len(self), 0, size=batch_size)  # type: np.ndarray
        if seq_len is not None:
            indices = np.array([np.arange(i, i + seq_len) for i in indices])
        assert isinstance(indices, np.ndarray)
        return self[indices]

    def append(self, x: X):
        if self.buffer is None:
            self.buffer = ArrayGroup.shape_like(x=x, pre_shape=(self.maxlen, ))
        stop = get_index(x)
        self[:stop] = x
        if self.pos + stop >= self.maxlen:
            self.full = True
        self.pos = self.modulate(stop)

    def extend(self, x: X):
        if self.buffer is None:
            self.buffer = ArrayGroup.shape_like(
                x=ArrayGroup(x)[0], pre_shape=(self.maxlen, ))

        self.append(x)
