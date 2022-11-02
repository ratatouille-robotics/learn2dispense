import gym
import yaml
import rospy
import pathlib
import numpy as np

from typing import Dict
from motion.commander import RobotMoveGroup
from learn2dispense.dispense_rollout import Dispenser


class Environment:
    """
    Class to maintain an internal state of the cell and continuously run with running into errors.
    Generates data samples for learning by interacting with the environment.
    """

    HOME = [-1.2334, -2.2579, 2.1997, -2.6269, -0.3113, 2.6590]

    MIN_DISPENSE_WEIGHT = 10
    MAX_DISPENSE_WEIGHT = 100
    REFILL_THRESHOLD = 10

    def __init__(self, log_dir: pathlib.Path) -> None:
        self.log_dir = log_dir

        self.container_wt = 0
        self.robot_mg = RobotMoveGroup()
        self.dispenser = Dispenser(self.robot_mg)
        self.container_wt = self.refill_container()

        assert self.robot_mg.go_to_joint_state(
            self.HOME, cartesian_path=True, velocity_scaling=0.15
        )

        # Load ingredient-specific params
        config_dir = pathlib.Path(__file__).parent.parent.parent
        with open(config_dir / "config/lentil_params.yaml") as f:
            self.ingredient_params = yaml.safe_load(f)

    def refill_container(self) -> float:
        user_input = float(input("Please refill container.\nEnter the weight of the container: "))
        _ = input("Press Enter to continue...")
        return float(user_input)

    def sample_weight(self) -> float:
        return np.random.uniform(self.MIN_DISPENSE_WEIGHT, min(self.MAX_DISPENSE_WEIGHT, self.container_wt))

    def interact(self, total_steps: int = None) -> Dict:
        """
        Will dispense for the requested timesteps and returns the requested data
        """
        data = {}
        current_steps = 0

        while(current_steps < total_steps):

            if(self.container_wt < self.REFILL_THRESHOLD):
                self.container_wt = self.refill_container()

            target_wt = self.sample_weight()
            rospy.loginfo(f"Requested wt: {target_wt:0.4f}")
            success, dispensed_wt, rollout_data = self.dispenser.dispense_ingredient(
                ingredient_params=self.ingredient_params,
                target_wt=target_wt
            )
            self.container_wt -= dispensed_wt

            if success:
                for k, v in rollout_data.items():
                    if k in rollout_data:
                        data[k].append(v)
                    else:
                        data[k] = [v]
                
                current_steps += len(rollout_data["state"])

        for k, v in data.items():
            data[k] = np.concatenate(v, axis=0)
        
        return data

    @property
    def observation_space(self) -> gym.spaces.Space:
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(13,),
            dtype=np.float32
        )

        return  observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        return action_space