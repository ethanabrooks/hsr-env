import glfw
import mujoco_py
import numpy as np

import hsr
from ppo.env_adapter import HSREnv
from ppo.main import add_hsr_args
from rl_utils import argparse, hierarchical_parse_args, space_to_size


class ControlViewer(mujoco_py.MjViewer):
    def __init__(self, sim):
        super().__init__(sim)
        self.active_joint = 0
        self.moving = False
        self.delta = None

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        keys = [
            glfw.KEY_0,
            glfw.KEY_1,
            glfw.KEY_2,
            glfw.KEY_3,
            glfw.KEY_4,
            glfw.KEY_5,
            glfw.KEY_6,
            glfw.KEY_7,
            glfw.KEY_8,
            glfw.KEY_9,
        ]
        if key in keys:
            self.active_joint = keys.index(key)
            print(self.sim.model.joint_names[self.active_joint])
        elif key == glfw.KEY_LEFT_CONTROL:
            self.moving = not self.moving
            self.delta = None

    def _cursor_pos_callback(self, window, xpos, ypos):
        if self.moving:
            self.delta = self._last_mouse_y - int(self._scale * ypos)
        super()._cursor_pos_callback(window, xpos, ypos)


class ControlHSREnv(HSREnv):
    def viewer_setup(self):
        self.viewer = ControlViewer(self.sim)

    def control_agent(self):
        action = np.zeros(space_to_size(self.action_space))
        action_scale = np.ones_like(action)
        action_scale[[0, 1]] = .1
        action[3] = 100
        if self.viewer and self.viewer.moving:
            print('delta =', self.viewer.delta)
        if self.viewer and self.viewer.moving and self.viewer.delta:
            action[self.viewer.active_joint] = self.viewer.delta
            # if self.sim.model.joint_names[self.viewer.active_joint] == 'l_proximal_joint':
            #     if action[self.sim.model.get_]
            print('delta =', self.viewer.delta)
            print('action =', action)

        s, r, t, i = self.step(action * action_scale)
        return t


def main(max_episode_steps, env_args):
    env = ControlHSREnv(**env_args)
    done = False

    action = np.zeros(space_to_size(env.action_space))
    action[0] = 1

    while True:
        if done:
            env.reset()
        done = env.control_agent()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_hsr_args(parser)
    hsr.util.env_wrapper(main)(**hierarchical_parse_args(parser))
