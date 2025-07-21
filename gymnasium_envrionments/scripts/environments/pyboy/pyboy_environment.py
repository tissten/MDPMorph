from functools import cached_property

import numpy as np
from environments.gym_environment import GymEnvironment
from util.configurations import GymEnvironmentConfig


import sys
import os

# 添加 pyboy_environment 的上一级目录（CARES_NEW）到模块搜索路径
sys.path.append(os.path.abspath("F:/UoA exention/Experiment/CARES_NEW/pyboy_environment"))
from pyboy_environment import suite
# from . import suite


class PyboyEnvironment(GymEnvironment):
    def __init__(self, config: GymEnvironmentConfig) -> None:
        super().__init__(config)

        self.env = suite.make(
            config.domain,
            config.task,
            config.act_freq,
            config.emulation_speed,
            config.headless,
        )

    @cached_property
    def min_action_value(self) -> float:
        return self.env.min_action_value

    @cached_property
    def max_action_value(self) -> float:
        return self.env.max_action_value

    @cached_property
    def observation_space(self) -> int:
        return self.env.observation_space

    @cached_property
    def action_num(self) -> int:
        return self.env.action_num

    def sample_action(self):
        return np.random.uniform(
            self.min_action_value, self.max_action_value, size=self.action_num
        )

    def set_seed(self, seed: int) -> None:
        self.env.set_seed(seed)

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: int) -> tuple:
        return self.env.step(action)

    def grab_frame(self, height=240, width=300) -> np.ndarray:
        return self.env.grab_frame(height, width)
