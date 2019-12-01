import glfw
import mujoco_py
import numpy as np
import hsr
from hsr.util import add_env_args
from rl_utils import argparse, hierarchical_parse_args, space_to_size
from operator import add
import sys
sys.path.append('/home/oidelima/ppo/ppo')
print("PATH ", sys.path)
from main import cli





def main(env_args):
    env = hsr.HSREnv(**env_args)
    cli()
    print("done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    wrapper_parser = parser.add_argument_group('wrapper_args')
    env_parser = parser.add_argument_group('env_args')
    hsr.util.add_env_args(env_parser)
    hsr.util.add_wrapper_args(wrapper_parser)
    args = hierarchical_parse_args(parser)
    print(repr(args)) 
    main_ = hsr.util.env_wrapper(main)(**args)
