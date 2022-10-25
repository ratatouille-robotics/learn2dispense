import gym
import numpy as np

from typing import Dict


class Environment:
    """
    Replacement for Ratatouille Planner

    The RL algo will request for a number of steps/episodes of dispensing

    Should maintain an internal state of the cell and continuously run with running into errors
    """

    def __init__(self) -> None:
        pass

    def interact(self, episodes: int = None, steps: int = None) -> Dict:
        """
        Will dispense for either a number of episodes or timesteps (only one will be provided)
        Will return the relevant data. The relavant data can be decided later
        """
        pass

    @property
    def observation_space(self) -> gym.spaces.Space:
        observation_space = gym.spaces.Box(
            low=13 * [-np.inf, ],
            high=13 * [np.inf, ], 
            shape=(13,),
            dtype=np.float32
        )

        return  observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        return action_space
