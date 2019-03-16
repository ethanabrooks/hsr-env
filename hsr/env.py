# stdlib
from collections import namedtuple
from pathlib import Path
from typing import List, Dict

import numpy as np
# third party
from gym import Space
from gym.spaces import Box

from hsr.mujoco_env import MujocoEnv


def get_xml_filepath(xml_filename=Path('models/world.xml')):
    return Path(Path(__file__).parent, xml_filename).absolute()


GoalSpec = namedtuple('GoalSpec', 'a b distance')


class HSREnv(MujocoEnv):
    def __init__(self,
                 xml_file: Path,
                 goals: List[GoalSpec],
                 starts: Dict[str, Box],
                 steps_per_action: int = 300,
                 obs_type: str = None,
                 render: bool = False,
                 record: bool = False,
                 record_freq: int = None,
                 render_freq: int = None,
                 ):
        self.starts = starts
        self.goals_specs = goals
        self.goals = None
        self._time_steps = 0
        if not xml_file.is_absolute():
            xml_file = get_xml_filepath(xml_file)

        self._obs_type = obs_type

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None

        self._record = record or record_freq
        self._render = render or render_freq
        self.render_freq = render_freq or 20
        self.record_freq = record_freq or 20
        self.steps_per_action = steps_per_action
        self._block_name = 'block'
        self._finger_names = ['hand_l_distal_link',
                              'hand_r_distal_link']
        super().__init__(str(xml_file), frame_skip=self.record_freq)
        self.initial_state = self.sim.get_state()

    def _get_observation(self):
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
            obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel])
        return obs

    def step(self, action):
        self.sim.data.ctrl[:] = action
        for i in range(self.steps_per_action):
            if self._render and i % self.render_freq == 0:
                self.render()
            if self._record and i % self.record_freq == 0:
                self.record()
            self.sim.step()
        self._time_steps += 1
        done = success = all([self.in_range(*s) for s in self.goals])
        reward = float(success)
        info = {'log count': {'success': success and self._time_steps > 0}}
        return self._get_observation(), reward, done, info

    def in_range(self, a, b, distance):
        pos1 = a if isinstance(a, np.ndarray) else self.sim.data.get_body_xpos(a)
        pos2 = b if isinstance(b, np.ndarray) else self.sim.data.get_body_xpos(b)
        return distance_between(pos1, pos2) < distance

    def reset(self):
        self._time_steps = 0
        self.sim.reset()
        state = self.sim.get_state()

        def sample_from_spaces(a, b, distance):
            if isinstance(a, Space):
                a = a.sample()
            if isinstance(b, Space):
                b = b.sample()
            return GoalSpec(a, b, distance)

        self.goals = [sample_from_spaces(*s) for s in self.goals_specs]
        self.sim.data.mocap_pos[:] = np.concatenate(
            [x for s in self.goals for x in [s.a, s.b]
             if isinstance(x, np.ndarray)])

        for joint, space in self.starts.items():
            assert isinstance(space, Space)
            start, end = self.model.get_joint_qpos_addr(joint)
            state.qpos[start: end] = space.sample()
        self.sim.set_state(state)
        self.sim.forward()
        return self._get_observation()

    def block_pos(self):
        return self.sim.data.get_body_xpos(self._block_name)

    def gripper_pos(self):
        finger1, finger2 = [
            self.sim.get_body_xpos(name) for name in self._finger_names
        ]
        return (finger1 + finger2) / 2.

    # def reset_recorder(self, record_path: Path):
    #     record_path.mkdir(parents=True, exist_ok=True)
    #     print(f'Recording video to {record_path}.mp4')
    #     video_recorder = VideoRecorder(
    #         env=self,
    #         base_path=str(record_path),
    #         metadata={'episode': self._episode},
    #         enabled=True,
    #     )
    #     closer.Closer().register(video_recorder)
    #     return video_recorder

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
