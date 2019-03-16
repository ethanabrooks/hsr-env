from pathlib import Path

import numpy as np
from gym.spaces import Box

from hsr.env import HSREnv

if __name__ == '__main__':
    env = HSREnv(xml_file=Path('models/world.xml'),
                 block_space=Box(
                     low=np.array([0, 0, .498]),
                     high=np.array([0, 0, .498]),
                 ),
                 min_lift_height=.08,
                 render=True)
    env.reset()
    while True:
        action = env.action_space.sample()
        s, r, t, i = env.step(action)
        if t:
            env.reset()
