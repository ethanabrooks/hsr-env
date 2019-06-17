import numpy as np

from rl_utils.types import Shape


def cartesian_product(*arrays):
    """
    https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single
    -array-of-2d-points
    senderle's answer
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def onehot(idx, num_entries):
    x = np.zeros(np.shape(idx) + (num_entries,))
    ix = np.meshgrid(*[range(i) for i in np.shape(idx)], indexing='ij')
    x[tuple(ix) + (idx,)] = 1
    return x


def is_scalar(x):
    try:
        return np.shape(x) == ()
    except ValueError:
        return False


def get_size(x):
    if x is None:
        return 0
    if is_scalar(x):
        return 1
    return sum(map(get_size, x))


def assign_to_vector(x, vector: np.ndarray):
    try:
        dim = vector.size / vector.shape[-1]
    except ZeroDivisionError:
        return
    if is_scalar(x):
        x = np.array([x])
    if isinstance(x, np.ndarray):
        vector.reshape(x.shape)[:] = x
    else:
        sizes = np.array(list(map(get_size, x)))
        sizes = np.cumsum(sizes / dim, dtype=int)
        for _x, start, stop in zip(x, [0] + list(sizes), sizes):
            indices = [slice(None) for _ in vector.shape]
            indices[-1] = slice(start, stop)
            assign_to_vector(_x, vector[tuple(indices)])


def vectorize(x, shape: Shape = None):
    if isinstance(x, np.ndarray):
        return x

    size = get_size(x)
    vector = np.zeros(size)
    if shape:
        vector = vector.reshape(shape)

    assert isinstance(vector, np.ndarray)
    assign_to_vector(x=x, vector=vector)
    return vector


def softmax(X, theta=1.0, axis=None):
    """
    Courtesy of https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    :param axis:
    """
    X = np.array(X)

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
