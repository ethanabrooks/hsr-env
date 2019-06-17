from typing import Callable, Iterable, Tuple

import gym
from gym import spaces
import numpy as np


def get_env_attr(env: gym.Env, attr: str):
    return getattr(unwrap_env(env, lambda e: hasattr(e, attr)), attr)


def unwrap_env(env: gym.Env, condition: Callable[[gym.Env], bool]):
    while not condition(env):
        try:
            env = env.env
        except AttributeError:
            raise RuntimeError(
                f"env {env} has no children that meet condition.")
    return env


def get_space_attrs(space: gym.Space, attr: str):
    if hasattr(space, attr):
        return getattr(space, attr)
    elif isinstance(space, gym.spaces.Dict):
        return {k: get_space_attrs(v, attr) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        return [get_space_attrs(s, attr) for s in space.spaces]
    raise RuntimeError(f'{space} does not have attribute {attr}.')


def concat_spaces(spaces: Iterable[gym.Space], axis: int = -1):
    def get_high_or_low(space: gym.Space, high: bool):
        if isinstance(space, gym.spaces.Box):
            return space.high if high else space.low
        if isinstance(space, gym.spaces.Dict):
            subspaces = space.spaces.values()
        elif isinstance(space, gym.spaces.Tuple):
            subspaces = space.spaces
        else:
            raise NotImplementedError
        concatted = concat_spaces(subspaces, axis=axis)
        return concatted.high if high else concatted.low

    def concat(high: bool):
        subspaces = [get_high_or_low(space, high=high) for space in spaces]
        return np.concatenate(subspaces, axis=axis)

    return gym.spaces.Box(high=concat(high=True), low=concat(high=False))


def space_shape(space: gym.Space):
    if isinstance(space, gym.spaces.Box):
        return space.low.shape
    if isinstance(space, gym.spaces.Dict):
        return {k: space_shape(v) for k, v in space.spaces.items()}
    if isinstance(space, gym.spaces.Tuple):
        return tuple(space_shape(s) for s in space.spaces)
    if isinstance(space, gym.spaces.Discrete):
        return (space.n, )
    raise NotImplementedError


def space_rank(space: gym.Space):
    def _rank(shape):
        if len(shape) == 0:
            return 0
        if isinstance(shape[0], int):
            for n in shape:
                assert isinstance(n, int)
            return len(shape)
        if isinstance(shape, dict):
            return {k: _rank(v) for k, v in shape.items()}
        if isinstance(shape, tuple):
            return tuple(_rank(s) for s in shape)

    return _rank(space_shape(space))


def space_to_size(space: gym.Space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple)):
        if isinstance(space, gym.spaces.Dict):
            _spaces = list(space.spaces.values())
        else:
            _spaces = list(space.spaces)
        return sum(space_to_size(s) for s in _spaces)
    else:
        return space.shape[0]


def make_box(*tuples: Tuple[float, float]):
    low, high = map(np.array, zip(*[(map(float, m)) for m in tuples]))
    return spaces.Box(low=low, high=high, dtype=np.float32)
