# stdlib
from collections import namedtuple
from pathlib import Path
from typing import Dict, List
import time

# third party
from gym import Space
from gym.spaces import Box
from gym.utils import closer
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np

from hsr.mujoco_env import MujocoEnv
from operator import add

def get_xml_filepath(xml_filename=Path('models/world.xml')):
    return Path(Path(__file__).parent, xml_filename).absolute()


GoalSpec = namedtuple('GoalSpec', 'a b distance')


class HSREnv(MujocoEnv):
    def __init__(
            self,
            xml_file: Path,
            goals: List[GoalSpec],
            starts: Dict[str, Box],
            steps_per_action: int = 300,
            obs_type: str = None,
            render: bool = False,
            record: bool = False,
            record_freq: int = None,
            render_freq: int = None,
            record_path: Path = None,
    ):
        self.starts = starts

        #KEYBOARD ROBOT CONTROL

        self.action = -1
        self.robot_speed = 0.007
        self.claw_rotation_speed = 0.03
        self.mocap_limits = {"bottom": 0.37, "back": -0.44, "front": 0.75, "left": 0.55, "right": -.55, "up":1.5}       
        self.guiding_mocap_pos = [-0.25955956,  0.00525669,  0.78973095] # Initial position of hand_palm_link
        self.claws_open = 0 # Control for the claws. Open --> 1, Closed --> 0
        self.claw_rotation_ctrl = 0 # -3.14 --> -90 degrees, 3.14 --> 90 degrees)
        self.color_goals = ["red", "blue", "white", "green"]
        self.goals_specs = goals
        self.goals = None
        self.goal = None
        self._time_steps = 0
        if not xml_file.is_absolute():
            xml_file = get_xml_filepath(xml_file)

        self._obs_type = obs_type

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None

        self.video_recorder = None
        self._record = any([record, record_path, record_freq])
        self._render = any([render, render_freq])
        self.record_freq = record_freq or 20
        self.render_freq = render_freq or 20
        record_path = record_path or '/tmp/training-video'
        self.steps_per_action = steps_per_action
        self._block_name = 'block0'
        self._finger_names = ['hand_l_distal_link', 'hand_r_distal_link']

        self.observation = None
        self.reward = None
        
        
        

        if self._record:
            self.video_recorder = VideoRecorder(
                env=self,
                base_path=record_path,
                enabled=True,
            )

        super().__init__(str(xml_file), frame_skip=self.record_freq)
        self.block_num = self.get_block_num()
        self.initial_state = self.sim.get_state()

        

    def _get_observation(self):

        #positions and orientations
        
        mocap_pos = self.sim.data.mocap_pos[1]
        block_pos = np.array([self.sim.data.get_body_xpos(body_name) for 
            body_name in self.sim.model.body_names if "block" in body_name])
        block_quat = np.array([self.sim.data.get_body_xquat(body_name) for 
            body_name in self.sim.model.body_names if "block" in body_name])
        grip_pos = self.gripper_pos()
        grip_ang_pos = self.sim.data.ctrl[:][self.sim.model.actuator_names.index('wrist_roll_motor')]
        
        if self.sim.data.ctrl[:][self.sim.model.actuator_names.index('hand_l_proximal_motor')] == 1 :
            grip_state = 1
        else:
            grip_state = 0

        block_pos = np.array([self.sim.data.get_body_xpos(body_name) for
            body_name in self.sim.model.body_names if "block" in body_name])
        
        left_finger_pos = self.sim.data.get_body_xpos('hand_l_finger_tip_frame')
        right_finger_pos = self.sim.data.get_body_xpos('hand_r_finger_tip_frame')
        fingers_pos = (left_finger_pos + right_finger_pos)/2

        #obs = np.array([*grip_pos.tolist(), grip_ang_pos, grip_state, *block_pos.tolist()[0], *block_quat.tolist()[0], *mocap_pos.tolist()])
        obs = np.array([*grip_pos.tolist(),*block_pos.tolist()[0]])
        #print("Gripper position: ", grip_pos.tolist())
        #print("Block position: ", block_pos.tolist()[0])
        #print("Mocap position: ", mocap_pos.tolist())
        return obs

    def step(self, action,  steps=None):

        """
        Actions:

            0 -> move forward
            1 -> move backwards
            2 -> move right
            3 -> move left
            4 -> move up
            5 -> move down
            6 -> rotate claws clockwise
            7 -> rotate claws counterclockwise
            8 -> open claws
            9 -> close claws


        """
        if action == 0:
            if self.guiding_mocap_pos[2] < self.mocap_limits["up"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [0.00, 0.00, self.robot_speed]) )
        elif action == 1:
            if self.guiding_mocap_pos[2] > self.mocap_limits["bottom"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [0.00, 0.00 , -self.robot_speed]) )
        elif action == 2:
            if self.guiding_mocap_pos[0] < self.mocap_limits["front"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [self.robot_speed, 0.00, 0.00]) )
        elif action == 3:
            if self.guiding_mocap_pos[0] > self.mocap_limits["back"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [-self.robot_speed, 0.00, 0.00]) )
        elif action == 4:
            if self.guiding_mocap_pos[1] < self.mocap_limits["left"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [0.00, self.robot_speed, 0.00]) )
        elif action == 5:
            if self.guiding_mocap_pos[1] > self.mocap_limits["right"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [0.00, -self.robot_speed, 0.00]) )


        #update claw rotation from action
        
        """if action == 0:
            if self.guiding_mocap_pos[0] < self.mocap_limits["front"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [self.robot_speed, 0.00, 0.00]) )
        elif action == 1:
            if self.guiding_mocap_pos[0] > self.mocap_limits["back"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [-self.robot_speed, 0.00, 0.00]) )
        elif action == 2:
            if self.guiding_mocap_pos[1] < self.mocap_limits["left"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [0.00, self.robot_speed, 0.00]) )
        elif action == 3:
            if self.guiding_mocap_pos[1] > self.mocap_limits["right"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [0.00, -self.robot_speed, 0.00]) )
        elif action == 4:
            if self.guiding_mocap_pos[2] < self.mocap_limits["up"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [0.00, 0.00, self.robot_speed]) )
        elif action == 5:
            if self.guiding_mocap_pos[2] > self.mocap_limits["bottom"]:
                self.guiding_mocap_pos = list( map(add, self.guiding_mocap_pos, [0.00, 0.00 , -self.robot_speed]) )
        elif  action == 7:
            if self.claw_rotation_ctrl > -3.14:
                self.claw_rotation_ctrl -= self.claw_rotation_speed
        elif action == 6:
            if self.claw_rotation_ctrl < 3.14:
                self.claw_rotation_ctrl += self.claw_rotation_speed
        elif action == 8:
            self.claws_open = 1
        elif action == 9:
            self.claws_open = -1
        elif action == 10:
            #do nothing
            pass"""

        self.sim.data.ctrl[:] = [0, 0, 0, self.claw_rotation_ctrl, self.claws_open, self.claws_open] #updates gripper rotation and open/closed state

        self.sim.data.mocap_pos[1] = self.guiding_mocap_pos 
        steps = steps or self.steps_per_action
        
        for i in range(steps):
            if self._render and i % self.render_freq == 0:
                self.render()
            if self._record and i % self.record_freq == 0:
                self.video_recorder.capture_frame()
            # Try to see if the simulation doesn't become unstable
            try:
                self.sim.step()
            except:
                self.reset_model()
            
        self._time_steps += 1
        self.reward = self._get_reward(self.goal)
        self.observation  = self._get_observation()
        

        block_pos = np.array([self.sim.data.get_body_xpos(body_name) for
            body_name in self.sim.model.body_names if "block" in body_name])

        left_finger_pos = self.sim.data.get_body_xpos('hand_l_finger_tip_frame')
        right_finger_pos = self.sim.data.get_body_xpos('hand_r_finger_tip_frame')
        fingers_pos = (left_finger_pos + right_finger_pos)/2
        distance = distance_between(fingers_pos, block_pos[0])

        done = distance < 0.09 or self._time_steps >= 128
        #done = self.reward == 1 or self.reward == -1 or self._time_steps > 100
        success = self.reward == 1 
        
        #info = {'log count': {'success': success and self._time_steps > 0}}
        return self.observation, self.reward, done, {}

    def _get_reward(self, goal):
        
        
        #reward = -0.1

        d = self.unwrapped.data

        block_pos = np.array([self.sim.data.get_body_xpos(body_name) for 
            body_name in self.sim.model.body_names if "block" in body_name])

        #if block falls off, penalize
        #for i in block_pos:
        #    if i[2] < 0.37: return -1

        #if  gripper gets close to block reward

        #grip_pos = self.gripper_pos()
        #grip_pos = grip_pos.tolist()
        left_finger_pos = self.sim.data.get_body_xpos('hand_l_finger_tip_frame')
        right_finger_pos = self.sim.data.get_body_xpos('hand_r_finger_tip_frame')
        fingers_pos = (left_finger_pos + right_finger_pos)/2
        distance = distance_between(fingers_pos, block_pos[0])

        #block_height = block_pos[0][2] - 0.422       
        #reward = -10 * distance #+ 100 * block_height
        #print(block_pos[0][2])
        #print(np.array([body_name for 
        #    body_name in self.sim.model.body_names]))
        #for coni in range(d.ncon):
        #    con = self.sim.data.contact[coni]
        #
        #    #if there is contact with table
        #    if (self.sim.model.geom_id2name(con.geom1) in fingers and
        #        "block" in self.sim.model.geom_id2name(con.geom2)):
        #        #if contact with multiple colors, penalize
        #        goal_bonus = 10

        reward = -distance

        #goal_bonus = 0
        #reward = 0.0
        #if distance < 0.07:
        #    reward = 1.0
            
        #reward = -10*distance #+ goal_bonus
        #self.distance = distance
        #print("DISTANCE: ", distance)
        """
        for block in self.target_blocks:
            for coni in range(d.ncon):
                con = self.sim.data.contact[coni]
            
                #if there is contact with table
                if (self.sim.model.geom_id2name(con.geom1) in self.color_goals and
                    self.sim.model.geom_id2name(con.geom2) == block):
                    #if contact with multiple colors, penalize
                        break
                    else: 
                        reward = 1
            if reward == 1: return reward
        """

        return reward



    def in_range(self, a, b, distance):
        def parse(x):
            if callable(x):
                return x()
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, str):
                return self.sim.data.get_body_xpos(x)
            raise RuntimeError(f"{x} must be function, np.ndarray, or string")

        return distance_between(parse(a), parse(b)) < distance

    def new_state(self):
        state = self.sim.get_state()
        for joint, space in self.starts.items():
            assert isinstance(space, Space)
            start, end = self.model.get_joint_qpos_addr(joint)


    def in_range(self, a, b, distance):
        def parse(x):
            if callable(x):
                return x()
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, str):
                return self.sim.data.get_body_xpos(x)
            raise RuntimeError(f"{x} must be function, np.ndarray, or string")

        return distance_between(parse(a), parse(b)) < distance

    def new_state(self):
        state = self.sim.get_state()
        for joint, space in self.starts.items():
            assert isinstance(space, Space)
            start, end = self.model.get_joint_qpos_addr(joint)
            state.qpos[start:end] = space.sample()

        return state

    def reset_model(self, init=False):
        self._time_steps = 0

        self.guiding_mocap_pos = [-0.25955956,  0.00525669,  0.78973095] # Initial position of hand_palm_link
        self.claws_open = 0 
        self.claw_rotation_ctrl = 0
   
        #reseting blocks positions
        if init == False:
            for joint_name in self.sim.model.joint_names:
                if "block" in joint_name:
                    i = self.sim.model.get_joint_qpos_addr(joint_name)[0]
                    #self.sim.data.qpos[i:i+7] = [0.32 * np.random.random() - 0.16, 
                    #    0.48 * np.random.random() - 0.24,0.422, np.random.random(), 0, 0, np.random.random()] 
                    self.sim.data.qpos[i:i+7] = [.8,.12,0.422, 0, 0, 0, 0] 

     
        
        state = self.new_state()
        self.sim.set_state(state)
        self.sim.forward()
        self.goal = self.get_new_goal()
        self.target_blocks = self.get_target_blocks(self.goal)
 
        self.observation = self._get_observation()
        return self.observation

    def get_target_blocks(self, goal):
        d = self.unwrapped.data
        target_blocks = []
        for coni in range(d.ncon):
            con = self.sim.data.contact[coni]
            
            #if there is contact with table
            if (self.sim.model.geom_id2name(con.geom1) == goal[0] and
                "block" in self.sim.model.geom_id2name(con.geom2)):
                target_blocks.append(self.sim.model.geom_id2name(con.geom2))

        target_blocks = list(dict.fromkeys(target_blocks))
        return target_blocks

    def get_new_goal(self):


        d = self.unwrapped.data
        goal_end = None
        while goal_end == None:
            goal_start = self.color_goals[np.random.randint(0,4)]
            for coni in range(d.ncon):
                con = self.sim.data.contact[coni]
                if (self.sim.model.geom_id2name(con.geom1) == goal_start and
                    "block" in self.sim.model.geom_id2name(con.geom2)):
                    goal_end = self.color_goals[np.random.randint(0,4)]
                    while goal_end == goal_start:
                        goal_end = self.color_goals[np.random.randint(0,4)]
                    break
        return [goal_start, goal_end]
                
                
    def get_block_color(self,block_num):
        rgba = [
            "0 1 0 1",
            "0 0 1 1",
            "0 1 1 1",
            "1 0 0 1",
            "1 0 1 1",
            "1 1 0 1",
            "1 1 1 1",
        ]
        if block_num == 0: return "Green"
        elif block_num == 1: return "Blue"
        elif block_num == 2: return "Aquamarine"
        elif block_num == 3: return "Red"
        elif block_num == 4: return "Purple"
        elif block_num == 5: return "Yellow"
        elif block_num == 6: return "White"

    def get_block_num(self):
        block_num = 0
        for body_name in self.sim.model.body_names:
            if "block" in body_name:
                block_num+=1
        return block_num

    def block_pos(self):
        return self.sim.data.get_body_xpos(self._block_name)

    def gripper_pos(self):
        finger1, finger2 = [
            self.sim.data.get_body_xpos(name) for name in self._finger_names
        ]
        return (finger1 + finger2) / 2.

    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        super().close()
        if self.video_recorder is not None:
            self.video_recorder.close()

    def reset_recorder(self, record_path: Path):
        record_path.mkdir(parents=True, exist_ok=True)
        print(f'Recording video to {record_path}.mp4')
        video_recorder = VideoRecorder(
            env=self,
            base_path=str(record_path),
            enabled=True,
        )
        closer.Closer().register(video_recorder)
        return video_recorder

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sim.__exit__()


def quaternion2euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    euler_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    euler_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    euler_z = np.arctan2(t3, t4)

    return euler_x, euler_y, euler_z


def distance_between(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))


def escaped(pos, world_upper_bound, world_lower_bound):
    # noinspection PyTypeChecker
    return np.any(pos > world_upper_bound) \
           or np.any(pos < world_lower_bound)


def get_limits(pos, size):
    return pos + size, pos - size


def point_inside_object(point, object):
    pos, size = object
    tl = pos - size
    br = pos + size
    return (tl[0] <= point[0] <= br[0]) and (tl[1] <= point[1] <= br[1])


def print1(*strings):
    print('\r', *strings, end='')


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
