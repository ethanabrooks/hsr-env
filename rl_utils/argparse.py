import argparse
from itertools import filterfalse
import re
from typing import List, Tuple

from gym import spaces
import numpy as np


def hierarchical_parse_args(parser: argparse.ArgumentParser,
                            include_positional=False):
    """
    :return:
    {
        group1: {**kwargs}
        group2: {**kwargs}
        ...
        **kwargs
    }
    """
    args = parser.parse_args()

    def key_value_pairs(group):
        for action in group._group_actions:
            if action.dest != 'help':
                yield action.dest, getattr(args, action.dest, None)

    def get_positionals(groups):
        for group in groups:
            if group.title == 'positional arguments':
                for k, v in key_value_pairs(group):
                    yield v

    def get_nonpositionals(groups: List[argparse._ArgumentGroup]):
        for group in groups:
            if group.title != 'positional arguments':
                children = key_value_pairs(group)
                descendants = get_nonpositionals(group._action_groups)
                yield group.title, {**dict(children), **dict(descendants)}

    positional = list(get_positionals(parser._action_groups))
    nonpositional = dict(get_nonpositionals(parser._action_groups))
    optional = nonpositional.pop('optional arguments')
    nonpositional = {**nonpositional, **optional}
    if include_positional:
        return positional, nonpositional
    return nonpositional


def parse_double(ctx, param, string):
    if string is None:
        return
    a, b = map(float, string.split(','))
    return a, b


def make_box(*tuples: Tuple[float, float]):
    low, high = map(np.array, zip(*[(map(float, m)) for m in tuples]))
    return spaces.Box(low=low, high=high, dtype=np.float32)


def parse_space(dim: int):
    def _parse_space(arg: str):
        regex = re.compile('\((-?[\.\d]+),(-?[\.\d]+)\)')
        matches = regex.findall(arg)
        if len(matches) != dim:
            raise argparse.ArgumentTypeError(
                f'Arg {arg} must have {dim} substrings '
                f'matching pattern {regex}.')
        return make_box(*matches)

    return _parse_space


def parse_vector(length: int, delim: str):
    def _parse_vector(arg: str):
        vector = tuple(map(float, arg.split(delim)))
        if len(vector) != length:
            raise argparse.ArgumentError(
                f'Arg {arg} must include {length} float values'
                f'delimited by "{delim}".')
        return vector

    return _parse_vector


def cast_to_int(arg: str):
    return int(float(arg))


try:
    import tensorflow as tf
    from rl_utils.tf import parametric_relu

    ACTIVATIONS = dict(
        relu=tf.nn.relu,
        leaky=tf.nn.leaky_relu,
        elu=tf.nn.elu,
        selu=tf.nn.selu,
        prelu=parametric_relu,
        sigmoid=tf.sigmoid,
        tanh=tf.tanh,
        none=None,
    )

    def parse_activation(arg: str):
        return ACTIVATIONS[arg]
except ImportError:
    pass
