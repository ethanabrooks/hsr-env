import subprocess

from rl_utils.argparse import *
from rl_utils.array_group import *
from rl_utils.gym import *
from rl_utils.numpy import *
from rl_utils.types import *
from rl_utils.replay_buffer import *

try:
    from rl_utils.tf import *
except ImportError:
    pass


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
