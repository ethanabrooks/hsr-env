from pathlib import Path

from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np

from hsr.env import GoalSpec, HSREnv

if __name__ == '__main__':
    env = HSREnv(
        xml_file=Path('models/world.xml'),
        goals=[GoalSpec(a='block', b=np.array([0, 0, .498]), distance=.05)],
        starts=dict(
            blockjoint=Box(
                #               x    y   z    q1 q2  q3 q4
                low=np.array([-.1, -.2, .418, 0, 0, -1, 0]),
                high=np.array([.1, +.2, .418, 1, 0, +1, 0]),
            )),
        render_freq=2)
    env = TimeLimit(env, max_episode_steps=20)
    env.reset()
    env.render()
    while True:
        action = env.action_space.sample()
        s, r, t, i = env.step(action)
        env.render()
        if t:
            env.reset()
