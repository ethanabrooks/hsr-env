# stdlib
from collections import namedtuple

# third party
from gym import spaces
import numpy as np

# first party
from environments.hindsight_wrapper import FrozenLakeHindsightWrapper, HindsightWrapper
from sac.utils import vectorize


class HierarchicalWrapper(HindsightWrapper):
    def goal_to_boss_action_space(self, goal: np.array):
        return goal

    def boss_action_to_goal_space(self, goal: np.array) -> np.ndarray:
        return goal


class FrozenLakeHierarchicalWrapper(HierarchicalWrapper,
                                    FrozenLakeHindsightWrapper):
    def __init__(self, env, n_boss_actions):
        super().__init__(env)
        fl = self.frozen_lake_env
        obs = super().reset()
        self._step = obs, fl.default_reward, False, {}

        self.observation_space = Hierarchical(
            boss=spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(np.shape(
                    vectorize([obs.achieved_goal, obs.desired_goal])))),
            worker=spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(np.shape(
                    vectorize([obs.observation, obs.desired_goal])))))

        self.action_space = Hierarchical(
            # DEBUG {{
            boss=spaces.Discrete(n_boss_actions),
            worker=spaces.Discrete(env.action_space.n))

    @property
    def boss_diameter(self):
        return int(np.sqrt(self.action_space.boss.n))

    @property
    def boss_radius(self):
        return self.boss_diameter // 2

    def goal_to_boss_action_space(self, goal: np.array):
        goal = np.minimum(goal, self.boss_radius * np.ones(2, dtype=int))
        goal = np.maximum(goal, self.boss_radius * -np.ones(2, dtype=int))
        min_goal = np.array([-self.boss_radius, -self.boss_radius])
        i, j = goal - min_goal
        action = np.zeros(self.action_space.boss.n)
        action[int(i * self.boss_diameter + j)] = 1
        return action

    def _boss_action_to_goal_space(self, action: int):
        n = np.sqrt(self.action_space.boss.n)
        return np.array([action // n, action % n]) - self.boss_radius

    def boss_action_to_goal_space(self, action: np.array):
        return self._boss_action_to_goal_space(np.argmax(action))


Hierarchical = namedtuple('Hierarchical', 'boss worker')
HierarchicalAgents = namedtuple('HierarchicalAgents',
                                'boss worker initial_state')
