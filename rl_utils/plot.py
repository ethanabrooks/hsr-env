#! /usr/bin/env python

# stdlib
import argparse
from pathlib import Path
from typing import List, Optional

# third party
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--names', nargs='*', type=Path)
    parser.add_argument('--paths', nargs='*', type=Path)
    parser.add_argument('--base-dir', default='.runs/logdir', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--tags', nargs='*')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--fname', type=str, default='plot')
    parser.add_argument('--quality', type=int)
    parser.add_argument('--dpi', type=int, default=256)
    main(**vars(parser.parse_args()))


def main(
        names: List[str],
        paths: List[Path],
        tags: List[str],
        tag: str,
        base_dir: Path,
        limit: Optional[int],
        quiet: bool,
        **kwargs,
):
    if tags and tag:
        raise RuntimeError('use either --tag or --tags, not both')
    if tag:
        tags = [tag] * len(names)

    if not (len(names) == len(paths) == len(tags)):
        raise RuntimeError(
            f'These three values should have the same number of values:\nnames: ({names})\npaths: ({paths})\ntags: ({tags})'
        )

    def get_tags():
        for name, path, tag in zip(names, paths, tags):
            path = Path(base_dir, path)
            if not path.exists():
                if not quiet:
                    print(f'{path} does not exist')

            for event_path in path.glob('**/events*'):
                print('Plotting', event_path)
                iterator = tf.train.summary_iterator(str(event_path))
                for event in iterator:
                    value = event.summary.value
                    if value:
                        if value[0].tag == tag:
                            value = value[0].simple_value
                            if limit is None or event.step < limit:
                                yield event.step, value, name

    header = tags[0] if len(tags) == 1 else ''
    data = pd.DataFrame(get_tags(), columns=['step', header, 'run'])
    sns.lineplot(x='step', y=header, hue='run', data=data)
    plt.legend(data['run'].unique())
    plt.axes().ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
    plt.savefig(**kwargs)


if __name__ == '__main__':
    cli()
