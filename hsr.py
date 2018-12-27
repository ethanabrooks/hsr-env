# stdlib
from collections import namedtuple
from contextlib import contextmanager
import itertools
from pathlib import Path
from typing import Tuple

# third party
from gym import spaces
from gym.utils import closer
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np

# first party
import mujoco
from mujoco import MujocoError, ObjType


def get_xml_filepath(xml_filename=Path('models/world.xml')):
    return Path(Path(__file__).parent, xml_filename).absolute()


class HSREnv:
    def __init__(self,
                 xml_file: Path,
                 steps_per_action: int,
                 geofence: float,
                 goal_space: spaces.Box,
                 block_space: spaces.Box,
                 min_lift_height: float = None,
                 randomize_pose: bool = False,
                 obs_type: str = None,
                 image_dims: Tuple[int] = None,
                 render: bool = False,
                 record_path: Path = None,
                 record_freq: int = None,
                 record: bool = False,
                 record_separate_episodes: bool = False,
                 render_freq: int = None,
                 no_random_reset: bool = False,
                 random_goals: bool = True):
        self.random_goals = random_goals
        if not xml_file.is_absolute():
            xml_file = get_xml_filepath(xml_file)

        self.no_random_reset = no_random_reset
        self.geofence = geofence
        self._obs_type = obs_type
        self._block_name = 'block'
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [
            left_finger_name,
            left_finger_name.replace('_l_', '_r_')
        ]
        self._episode = 0
        self._time_steps = 0

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None
        self.render_freq = 20 if (render
                                  and render_freq is None) else render_freq
        self.steps_per_action = steps_per_action

        # record stuff
        self._video_recorder = None
        self._record_separate_episodes = record_separate_episodes
        self._record = any((record_separate_episodes, record_path, record_freq,
                            record))
        if self._record:
            self._record_path = record_path or Path('/tmp/training-video')
            image_dims = image_dims or (1000, 1000)
            self._record_freq = record_freq or 20

            if not record_separate_episodes:
                self._video_recorder = self.reset_recorder(self._record_path)
        else:
            image_dims = image_dims or []
        self._image_dimensions = image_dims

        self.sim = mujoco.Sim(str(xml_file), *image_dims, n_substeps=1)

        # initial values
        self.initial_qpos = self.sim.qpos.ravel().copy()
        self.initial_qvel = self.sim.qvel.ravel().copy()

        # block stuff
        self.initial_block_pos = []
        self._block_qposadrs = []
        self.n_blocks = 0
        offset = np.array([
            0,  # x
            1,  # y
            3,  # quat0
            6,  # quat3
        ])
        self._block_space = block_space
        try:
            for i in itertools.count():
                self.initial_block_pos.append(
                    np.copy(self.block_pos(blocknum=i)))
                joint_offset = self.sim.get_jnt_qposadr(
                    f'block{i}joint') + offset
                self._block_qposadrs.append(joint_offset)
                self.n_blocks = i + 1
        except MujocoError:
            pass

        # goal space
        self._min_lift_height = min_lift_height
        self.goal_space = goal_space

        epsilon = .0001
        too_close = self.goal_space.high - self.goal_space.low < 2 * epsilon
        self.goal_space.high[too_close] += epsilon
        self.goal_space.low[too_close] -= epsilon
        self.goal = None

        def using_joint(name):
            return self.sim.contains(ObjType.JOINT, name)

        self._base_joints = list(filter(using_joint, ['slide_x', 'slide_y']))
        if obs_type == 'openai':
            raw_obs_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(25, ),
                dtype=np.float32,
            )
        else:
            raw_obs_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.sim.nq + len(self._base_joints), ),
                dtype=np.float32,
            )
        self.observation_space = spaces.Tuple(
            Observation(observation=raw_obs_space, goal=self.goal_space))

        # joint space
        all_joints = [
            'slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint',
            'wrist_roll_joint', 'hand_l_proximal_joint'
        ]
        self._joints = list(filter(using_joint, all_joints))
        jnt_range_idx = [
            self.sim.name2id(ObjType.JOINT, j) for j in self._joints
        ]
        self._joint_space = spaces.Box(
            *map(np.array, zip(*self.sim.jnt_range[jnt_range_idx])),
            dtype=np.float32)
        self._joint_qposadrs = [
            self.sim.get_jnt_qposadr(j) for j in self._joints
        ]
        self.randomize_pose = randomize_pose

        # action space
        self.action_space = spaces.Box(
            low=self.sim.actuator_ctrlrange[:-1, 0],
            high=self.sim.actuator_ctrlrange[:-1, 1],
            dtype=np.float32)

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, mode=None, camera_name=None, labels=None):
        if mode == 'rgb_array':
            return self.sim.render_offscreen(
                camera_name=camera_name, labels=labels)
        return self.sim.render(camera_name=camera_name, labels=labels)

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(camera_name)

    def compute_terminal(self):
        return self.is_successful()

    def compute_reward(self):
        if self.is_successful():
            return 1
        else:
            return 0

    def _get_obs(self):
        if self._obs_type == 'openai':

            # positions
            grip_pos = self.gripper_pos()
            dt = self.sim.nsubsteps * self.sim.timestep
            object_pos = self.block_pos()
            grip_velp = .5 * sum(
                self.sim.get_body_xvelp(name) for name in self._finger_names)
            # rotations
            object_rot = mat2euler(self.sim.get_body_xmat(self._block_name))

            # velocities
            object_velp = self.sim.get_body_xvelp(self._block_name) * dt
            object_velr = self.sim.get_body_xvelr(self._block_name) * dt

            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
            gripper_state = np.array([
                self.sim.get_joint_qpos(f'hand_{x}_proximal_joint')
                for x in 'lr'
            ])
            qvels = np.array([
                self.sim.get_joint_qvel(f'hand_{x}_proximal_joint')
                for x in 'lr'
            ])
            gripper_vel = dt * .5 * qvels

            obs = np.concatenate([
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ])
        else:
            base_qvels = [
                self.sim.get_joint_qvel(j) for j in self._base_joints
            ]
            obs = np.concatenate([self.sim.qpos, base_qvels])
        observation = Observation(observation=obs, goal=self.goal)
        # assert self.observation_space.contains(observation)
        return observation

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        mirrored = 'hand_l_proximal_motor'
        mirroring = 'hand_r_proximal_motor'

        # insert mirrored values at the appropriate indexes
        mirrored_index, mirroring_index = [
            self.sim.name2id(ObjType.ACTUATOR, n)
            for n in [mirrored, mirroring]
        ]
        # necessary because np.insert can't append multiple values to end:
        mirroring_index = np.minimum(mirroring_index, self.action_space.shape)
        action = np.insert(action, mirroring_index, action[mirrored_index])

        self._time_steps += 1
        assert np.shape(action) == np.shape(self.sim.ctrl)

        assert np.shape(action) == np.shape(self.sim.ctrl)
        for i in range(self.steps_per_action):
            self.sim.ctrl[:] = action
            self.sim.step()
            if self.render_freq is not None and i % self.render_freq == 0:
                self.render()
            if self._record and i % self._record_freq == 0:
                self._video_recorder.capture_frame()

        done = np.squeeze(self.compute_terminal())
        reward = np.squeeze(self.compute_reward())

        # pause when goal is achieved
        if reward > 0:
            for _ in range(50):
                if self.render_freq is not None:
                    self.render()
                if self._record:
                    self._video_recorder.capture_frame()

        info = {'log count': {'success': reward > 0 and self._time_steps > 1}}
        return self._get_obs(), reward, done, info

    def _sync_grippers(self, qpos):
        qpos[self.sim.get_jnt_qposadr('hand_r_proximal_joint')] = qpos[
            self.sim.get_jnt_qposadr('hand_l_proximal_joint')]

    def reset_sim(self, qpos: np.ndarray):
        assert qpos.shape == (self.sim.nq, )
        self.initial_qpos = qpos
        self.sim.qpos[:] = qpos.copy()
        self.sim.qvel[:] = 0
        self.sim.forward()

    @contextmanager
    def get_initial_qpos(self):
        qpos = np.copy(self.initial_qpos)
        yield qpos
        self.reset_sim(qpos)

    def reset(self):
        if self.no_random_reset:
            with self.get_initial_qpos() as qpos:
                for i, adrs in enumerate(self._block_qposadrs):

                    # if block out of bounds
                    if not self.goal_space.contains(self.block_pos(i)):
                        # randomize blocks
                        qpos[adrs] = self._block_space.sample()

        else:
            self.sim.reset()  # restore original qpos
            with self.get_initial_qpos() as qpos:
                if self.randomize_pose:
                    # randomize joint angles
                    qpos[self._joint_qposadrs] = self._joint_space.sample()
                    self._sync_grippers(qpos)

                # randomize blocks
                for adrs in self._block_qposadrs:
                    qpos[adrs] = self._block_space.sample()

        self.set_goal(self.new_goal())

        if self._time_steps > 0:
            self._episode += 1
        self._time_steps = 0

        # if necessary, reset VideoRecorder
        if self._record and self._record_separate_episodes:
            if self._video_recorder:
                self._video_recorder.close()
            record_path = Path(self._record_path, str(self._episode))
            self._video_recorder = self.reset_recorder(record_path)

        return self._get_obs()

    def new_goal(self):
        if self._min_lift_height:
            return self.block_pos() + np.array([0, 0, self._min_lift_height])
        elif self.random_goals or self.goal is None:
            return self.goal_space.sample()
        else:
            return self.goal

    def achieved_goal(self):
        return self.block_pos()

    def block_pos(self, blocknum=0):
        return self.sim.get_body_xpos(self._block_name + str(blocknum))

    def gripper_pos(self):
        finger1, finger2 = [
            self.sim.get_body_xpos(name) for name in self._finger_names
        ]
        return (finger1 + finger2) / 2.

    def reset_recorder(self, record_path: Path):
        record_path.mkdir(parents=True, exist_ok=True)
        print(f'Recording video to {record_path}.mp4')
        video_recorder = VideoRecorder(
            env=self,
            base_path=str(record_path),
            metadata={'episode': self._episode},
            enabled=True,
        )
        closer.Closer().register(video_recorder)
        return video_recorder

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sim.__exit__()

    def is_successful(self, achieved_goal=None, desired_goal=None):
        if self._min_lift_height:
            return self.block_pos(
            )[2] > self.initial_block_pos[0][2] + self._min_lift_height

        # only check first block
        if achieved_goal is None:
            achieved_goal = self.block_pos()
        if desired_goal is None:
            desired_goal = self.goal
        return distance_between(achieved_goal, desired_goal) < self.geofence

    def set_goal(self, goal: np.ndarray):
        # assert self.goal_space.contains(goal)
        self.sim.mocap_pos[:] = goal
        self.goal = goal


class MultiBlockHSREnv(HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.goals = None

    def is_successful(self, achieved_goal=None, desired_goal=None):
        if achieved_goal is None:
            achieved_goal = np.stack(
                [self.block_pos(i).copy() for i in range(self.n_blocks)])
        if desired_goal is None:
            desired_goal = self.goals

        return np.all(
            distance_between(achieved_goal[..., :2], desired_goal[..., :2]) <
            self.geofence,
            axis=-1)

    def set_goal(self, goal: np.ndarray):
        goal[2] = self.initial_block_pos[0][2]
        super().set_goal(goal)
        self.goals = np.stack(
            [self.goal] +
            [self.block_pos(i).copy() for i in range(1, self.n_blocks)])


class MoveGripperEnv(HSREnv):
    def is_successful(self, achieved_goal=None, desired_goal=None):
        if achieved_goal is None:
            achieved_goal = self.gripper_pos()
        if desired_goal is None:
            desired_goal = self.goal
        return super().is_successful(
            achieved_goal=achieved_goal, desired_goal=desired_goal)

    def achieved_goal(self):
        return self.gripper_pos()

    def compute_reward(self):
        return 0 if self.is_successful() else -1


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
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] +
                 mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > np.finfo(np.float64).eps * 4.
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition, -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0)
    return euler


Observation = namedtuple('Obs', 'observation goal')
