import glfw
import mujoco_py
import numpy as np
import hsr
from hsr.util import add_env_args
from rl_utils import argparse, hierarchical_parse_args, space_to_size
from pynput.keyboard import Key, Listener
from operator import add

ROBOT_SPEED = 0.005

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



class ControlHSREnv(hsr.HSREnv):
    
    def viewer_setup(self):
        self.viewer = ControlViewer(self.sim)

    def control_agent(self):

        #mocap_pos is updated by the keyboard event handler
        self.sim.data.mocap_pos[1] = mocap_pos

        action = np.zeros(space_to_size(self.action_space))
        action_scale = np.ones_like(action)
 
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


def main(env_args):
    
    env = ControlHSREnv(**env_args)
    done = False
   
    action = np.zeros(space_to_size(env.action_space))
    action[0] = 1
  
    while(True):
        if done:
            env.reset()
        done = env.control_agent()


#KEYBOARD ROBOT CONTROL
mocap_pos = [-0.25955956,  0.00525669,  0.78973095] #initial position of hand_palm_link

        
def on_press(key):
    try:
        global mocap_pos
        
        if key.char == 'o': 
            mocap_pos = list( map(add, mocap_pos, [ROBOT_SPEED, 0.00, 0.00]) )
        elif key.char == 'l': 
            mocap_pos = list( map(add, mocap_pos, [-ROBOT_SPEED, 0.00, 0.00]) )  
        elif key.char == 'k': 
            mocap_pos = list( map(add, mocap_pos, [0.00, ROBOT_SPEED, 0.00]) )
        elif key.char == ';': 
            mocap_pos = list( map(add, mocap_pos, [0.00, -ROBOT_SPEED, 0.00]) )
        elif key.char == 'u': 
            mocap_pos = list( map(add, mocap_pos, [0.00, 0.00, ROBOT_SPEED]) )
        elif key.char == 'j':  
            mocap_pos = list( map(add, mocap_pos, [0.00, 0.00 , -ROBOT_SPEED]) )        
        
    except AttributeError:
         print('special key {0} pressed'.format(
            key))

def on_release(key):

    if key == Key.esc:
        return False

listener = Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    wrapper_parser = parser.add_argument_group('wrapper_args')
    env_parser = parser.add_argument_group('env_args')
    hsr.util.add_env_args(env_parser)
    hsr.util.add_wrapper_args(wrapper_parser)
    args = hierarchical_parse_args(parser) 
    main_ = hsr.util.env_wrapper(main)(**args)
