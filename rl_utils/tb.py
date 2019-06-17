#! /usr/bin/env python

# stdlib
import argparse
from pathlib import Path
import subprocess


def cmd(args, fail_ok=False, cwd=None):
    process = subprocess.Popen(
        args,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=cwd,
        universal_newlines=True)
    stdout, stderr = process.communicate(timeout=1)
    if stderr and not fail_ok:
        raise RuntimeError(f"Command `{' '.join(args)}` failed: {stderr}")
    return stdout.strip()


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('port', type=int)
    parser.add_argument('path', type=Path)
    parser.add_argument('--logdir', type=Path, default=Path('.runs/logdir'))
    tb(**vars(parser.parse_args()))


def tb(port: int, path: Path, logdir: Path):
    active_sessions = cmd('tmux ls -F #{session_name}'.split())
    session_name = f'tensorboard{port}'
    logdir = Path(logdir, path)
    if not logdir.exists():
        raise RuntimeError(f'Path {logdir} does not exist.')
    command = f'tensorboard --logdir={logdir} --port={port}'
    if session_name in active_sessions:
        window_name = f'{session_name}:0'
        cmd(['tmux', 'respawn-window', '-t', window_name, '-k', command])
        print(f'Respawned {window_name} window.')
    else:
        cmd(['tmux', 'new', '-d', '-s', session_name, command])
        print(f'Created new session called {session_name}.')


if __name__ == '__main__':
    cli()
