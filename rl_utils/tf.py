import tensorflow as tf
from tensorflow.python import debug as tf_debug


def create_sess(debug=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.inter_op_parallelism_threads = 1
    sess = tf.Session(config=config)
    if debug:
        return tf_debug.LocalCLIDebugWrapperSession(sess)
    return sess


def make_network(input_size: int, output_size: int, n_hidden: int, layer_size: int,
                 activation, name='fc', **kwargs, ) \
        -> \
                tf.keras.Sequential:
    sizes = [layer_size] * n_hidden
    activations = [activation] * n_hidden + [None]
    return tf.keras.Sequential([
        tf.layers.Dense(
            input_shape=(in_size, ),
            units=out_size,
            activation=activation,
            name=f'{name}{i}',
            **kwargs) for i, (in_size, out_size, activation) in enumerate(
                zip(
                    [input_size] + sizes,
                    sizes + [output_size],
                    activations,
                ))
    ])


def parametric_relu(_x):
    alphas = tf.get_variable(
        'alpha',
        _x.get_shape()[-1],
        initializer=tf.constant_initializer(0.0),
        dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)
