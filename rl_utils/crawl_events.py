#! /usr/bin/env python

# stdlib
import argparse
import itertools
from collections import namedtuple
from itertools import islice
from pathlib import Path
from typing import List, Optional

# third party
import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dirs', nargs='*', type=Path)
    parser.add_argument(
        '--base-dir', default='.runs/logdir', type=Path, help=' ')
    parser.add_argument('--smoothing', type=int, default=10, help=' ')
    parser.add_argument('--tag', default='return', help=' ')
    parser.add_argument(
        '--no-cache-write', dest='write_cache', action='store_false', help=' ')
    parser.add_argument('--quiet', action='store_true', help=' ')
    parser.add_argument('--show-num-values', action='store_true', help=' ')
    parser.add_argument('--show-cache-writes', action='store_true', help=' ')
    parser.add_argument('--until-time', type=int, help=' ')
    parser.add_argument('--until-step', type=int, help=' ')
    main(**vars(parser.parse_args()))


def main(
        base_dir: Path,
        dirs: Path,
        tag: str,
        smoothing: int,
        write_cache: bool,
        show_num_values: bool,
        show_cache_writes: bool,
        quiet: bool,
        until_time: int,
        until_step: int,
):
    def get_event_files():
        for dir in dirs:
            yield from Path(base_dir, dir).glob('**/events*')

    def get_values(path):
        start_time = None
        iterator = tf.train.summary_iterator(str(path))
        for i in itertools.count():
            try:
                event = next(iterator)
                if start_time is None:
                    start_time = event.wall_time
                if until_time is not None and \
                        event.wall_time - start_time > until_time:
                    return
                if until_step is not None and event.step > until_step:
                    return
                for value in event.summary.value:
                    if value.tag == tag:
                        yield value.simple_value
            except DataLossError:
                if not quiet:
                    print('Data loss in', path)
            except StopIteration:
                return

    def get_averages():
        for path in get_event_files():
            iterator = get_values(path)
            num_values = sum(1 for _ in get_values(path))
            if show_num_values:
                print(f'Read {num_values} in {path}.')
            length = min(num_values,
                         smoothing)  # amount of data to actually use
            iterator = itertools.islice(iterator, num_values - length,
                                        num_values)
            if length > 0:
                yield sum(iterator) / length, path
            else:
                yield None

    averages = [x for x in get_averages() if x is not None]
    for data, path in averages:
        cache_path = Path(path.parent, f'{smoothing}.{tag}')
        if write_cache:
            if show_cache_writes:
                print(f'Writing {cache_path}...')
            with cache_path.open('w') as f:
                f.write(str(data))

    if not quiet:
        print('Sorted lowest to highest:')
        print('*************************')
        for data, path in sorted(averages):
            if data is None:
                print('No data found in', path)
            else:
                print('{:10}: {}'.format(data, path))


if __name__ == '__main__':
    cli()
