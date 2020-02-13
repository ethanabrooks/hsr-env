import glfw
import mujoco_py
import numpy as np
import hsr
from hsr.util import add_env_args
from rl_utils import argparse, hierarchical_parse_args, space_to_size
from operator import add





class ControlViewer(mujoco_py.MjViewer):
    def __init__(self, sim):
        super().__init__(sim)
        self.active_joint = 0
        self.moving = False
        self.delta = None
        

    def _create_full_overlay(self):
        super()._create_full_overlay()
        self.add_overlay(
            0, "Reset", "N")
        self.add_overlay(
            0, "Move forward", "O")
        self.add_overlay(
            0, "Move backward", "L")
        self.add_overlay(
            0, "Move left", "K")
        self.add_overlay(
            0, "Move right", ";")
        self.add_overlay(
            0, "Move up", "U")
        self.add_overlay(
            0, "Move down", "J")
        self.add_overlay(
            0, "Open claws", "P")
        self.add_overlay(
            0, "Close claws", "[")
        self.add_overlay(
            0, "Turn claws clockwise", "]")
        self.add_overlay(
            0, "Turn claws counter-clockwise", "\\")
        self.add_overlay(
            1, "Goal", self.env.goal[0].capitalize() +
             " to " + self.env.goal[1].capitalize())
        self.add_overlay(
            1, "Reward", str(self.env.reward))
        """self.add_overlay(
            1, "Gripper position", str(round(self.env.observation[0][0],2)) + 
            " " + str(round(self.env.observation[0][1],2)) + " " + 
            str(round(self.env.observation[0][2],2)))
        self.add_overlay(
            1, "Gripper angular position", str(round(self.env.observation[1],2)) + " rad")
        self.add_overlay(
            1, "Gripper state", self.env.observation[2].capitalize())
        for block_num in range(len(self.env.observation[3])):
            self.add_overlay(
                1, self.env.get_block_color(block_num) + " block position",
                str(round(self.env.observation[3][block_num][0],2)) + 
                " " + str(round(self.env.observation[3][block_num][1],2)) + " " + 
                str(round(self.env.observation[3][block_num][2],2)))
            self.add_overlay(
                1, self.env.get_block_color(block_num) + " block quaternion",
                str(round(self.env.observation[4][block_num][0],2)) + 
                " " + str(round(self.env.observation[4][block_num][1],2)) + " " + 
                str(round(self.env.observation[4][block_num][2],2)) + 
                str(round(self.env.observation[4][block_num][3],2)))"""
        self.add_overlay(
            1, "Mocap position", str(round(self.env.guiding_mocap_pos[0],2)) + 
            " " + str(round(self.env.guiding_mocap_pos[1],2)) + " " + 
            str(round(self.env.guiding_mocap_pos[2],2)))
       

        

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
        elif key == glfw.KEY_O:
            self.env.action[:3] = np.array([self.env.robot_speed,0,0])
        elif key == glfw.KEY_L:
            self.env.action[:3] = np.array([-self.env.robot_speed,0,0]) 
        elif key == glfw.KEY_K: 
            self.env.action[:3] = np.array([0, self.env.robot_speed,0])
        elif key == glfw.KEY_SEMICOLON:
            self.env.action[:3] = np.array([0, -self.env.robot_speed,0])
        elif key == glfw.KEY_U:  
            self.env.action[:3] = np.array([0,0,self.env.robot_speed])
        elif key == glfw.KEY_J: 
            self.env.action[:3] = np.array([0,0, -self.env.robot_speed])
        elif key == glfw.KEY_P:  
            self.env.action[3] = 1
            self.env.claws_open = 1
        elif key == glfw.KEY_LEFT_BRACKET:   
            self.env.action[3] = -1
            self.env.claws_open = -1
        elif key == glfw.KEY_RIGHT_BRACKET: 
            if self.env.claw_rotation_ctrl > -3.14:  
                self.env.claw_rotation_ctrl -= self.env.claw_rotation_speed
        elif key == glfw.KEY_BACKSLASH:
            if self.env.claw_rotation_ctrl < 3.14: 
                self.env.claw_rotation_ctrl += self.env.claw_rotation_speed
        elif key == glfw.KEY_N:
            self.env.reset()
        
       
       


    def _cursor_pos_callback(self, window, xpos, ypos):
        if self.moving:
            self.delta = self._last_mouse_y - int(self._scale * ypos)
        super()._cursor_pos_callback(window, xpos, ypos)



class ControlHSREnv(hsr.HSREnv):
    
    def viewer_setup(self):
        self.viewer = ControlViewer(self.sim)
        self.viewer.env = self

    def control_agent(self):
        s, r, t, i = self.step(self.action)          
        return t


def main(env_args):
    env = ControlHSREnv(**env_args)
    done = False
    env.reset_model(init = True)  
    reset_count = 0
    while(True):
        
        if done:
            reset_count+=1
            if reset_count == 20:
                env.reset()
                reset_count = 0
        done = env.control_agent()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    wrapper_parser = parser.add_argument_group('wrapper_args')
    env_parser = parser.add_argument_group('env_args')
    hsr.util.add_env_args(env_parser)
    hsr.util.add_wrapper_args(wrapper_parser)
    args = hierarchical_parse_args(parser) 
    main_ = hsr.util.env_wrapper(main)(**args)
