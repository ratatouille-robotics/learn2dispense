import gym
import yaml
import rospy
import pathlib
import numpy as np

from typing import Dict, Optional
from motion.commander import RobotMoveGroup
from motion.utils import offset_pose, make_pose
from stable_baselines3.common.policies import BasePolicy
from learn2dispense.dispense_rollout import Dispenser


class Environment:
    """
    Class to maintain an internal state of the cell and continuously run with running into errors.
    Generates data samples for learning by interacting with the environment.
    """
    HOME = [-0.5992, -2.4339, 2.2566, -2.9490, -1.0004, 3.0982]
    SHELF_FRONT = [-0.05481, -2.0645, 2.480, -3.2914, 0.0045, 2.8663]
    CONTAINER_SHELF_POSE_1 = [-0.144, -0.472, 0.477, 0.729, -0.004, -0.008, 0.685]
    CONTAINER_SHELF_POSE_2 = [-0.291, -0.472, 0.477, 0.729, -0.004, -0.008, 0.685]
    CONTAINER_SCALE_POSE = [-0.436, 0.029, 0.214, -0.498, 0.521, 0.498, -0.482]

    MIN_DISPENSE_WEIGHT = 10
    MAX_DISPENSE_WEIGHT = 100
    REFILL_THRESHOLD = 10
    EMPTY_CONTAINER_WEIGHT = 116.5

    def __init__(self, log_dir: pathlib.Path, pick_container_on_start: bool = True) -> None:
        self.log_dir = log_dir
        self.robot_mg = RobotMoveGroup()
        self.dispenser = Dispenser(self.robot_mg)
        self._pre_process()

        assert self.robot_mg.go_to_joint_state(self.HOME, cartesian_path=False)

        if pick_container_on_start:
            self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_1, return_home=True)
            self.place_container_on_scale()
            self.pick_container_from_scale()
            self.place_container_on_shelf(self.CONTAINER_SHELF_POSE_1)
            # To start with, pick and place empty container on weighing scale
            self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_2, return_home=True)
            self.place_container_on_scale()
            # Pick full container for dispensing
            self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_1, return_home=True)

        # Load ingredient-specific params
        config_dir = pathlib.Path(__file__).parent.parent.parent
        with open(config_dir / "config/lentil_params.yaml") as f:
            self.ingredient_params = yaml.safe_load(f)

    def _pre_process(self):
        self.CONTAINER_SCALE_POSE = make_pose(self.CONTAINER_SCALE_POSE[:3], self.CONTAINER_SCALE_POSE[3:])
        self.CONTAINER_SHELF_POSE_1 = make_pose(self.CONTAINER_SHELF_POSE_1[:3], self.CONTAINER_SHELF_POSE_1[3:])
        self.CONTAINER_SHELF_POSE_2 = make_pose(self.CONTAINER_SHELF_POSE_2[:3], self.CONTAINER_SHELF_POSE_2[3:])

    def pick_container_from_shelf(self, pick_pose, return_home: bool = False):
        # Open gripper
        self.robot_mg.open_gripper(wait=True)
        # Go to designated pose
        assert self.robot_mg.go_to_joint_state(self.SHELF_FRONT, cartesian_path=False)
        assert self.robot_mg.go_to_pose_goal(offset_pose(pick_pose, [0, 0.2, 0.0]), cartesian_path=False)
        assert self.robot_mg.go_to_pose_goal(pick_pose, wait=True)
        # Close gripper
        assert self.robot_mg.close_gripper(wait=True)
        # Retract
        assert self.robot_mg.go_to_pose_goal(offset_pose(pick_pose, [0, 0.0, 0.015]))
        assert self.robot_mg.go_to_pose_goal(offset_pose(pick_pose, [0, 0.2, 0.015]))
        if return_home:
            assert self.robot_mg.go_to_joint_state(self.HOME, cartesian_path=False)

    def place_container_on_shelf(self, place_pose, return_home: bool = False):
        # Go to designated pose
        assert self.robot_mg.go_to_joint_state(self.SHELF_FRONT, cartesian_path=False)
        assert self.robot_mg.go_to_pose_goal(offset_pose(place_pose, [0, 0.2, 0.015]), cartesian_path=False)
        assert self.robot_mg.go_to_pose_goal(offset_pose(place_pose, [0, 0, 0.015]))
        assert self.robot_mg.go_to_pose_goal(place_pose)
        # Open gripper
        self.robot_mg.open_gripper(wait=True)
        # Retract
        assert self.robot_mg.go_to_pose_goal(offset_pose(place_pose, [0, 0.2, 0]))
        if return_home:
            assert self.robot_mg.go_to_joint_state(self.HOME, cartesian_path=False)

    def pick_container_from_scale(self):
        init_weight = self.dispenser.get_weight()
        # Open gripper
        self.robot_mg.open_gripper(wait=True)
        # Go to designated pose
        assert self.robot_mg.go_to_pose_goal(offset_pose(self.CONTAINER_SCALE_POSE, [0.01, 0.0, 0.15]))
        assert self.robot_mg.go_to_pose_goal(offset_pose(self.CONTAINER_SCALE_POSE, [0.01, 0.0, 0]))
        assert self.robot_mg.go_to_pose_goal(self.CONTAINER_SCALE_POSE)
        # Close gripper
        assert self.robot_mg.close_gripper(wait=True)
        # Retract
        assert self.robot_mg.go_to_pose_goal(offset_pose(self.CONTAINER_SCALE_POSE, [0, 0.0, 0.15]))
        assert self.robot_mg.go_to_joint_state(self.HOME, cartesian_path=True)

        self.available_weight = init_weight - self.dispenser.get_weight() - self.EMPTY_CONTAINER_WEIGHT
        rospy.loginfo(f"Available ingredient quantity: {self.available_weight:0.2f} g")

    def place_container_on_scale(self):
        # Go to designated pose
        assert self.robot_mg.go_to_pose_goal(
            offset_pose(self.CONTAINER_SCALE_POSE, [0, 0, 0.15]), cartesian_path=False
        )
        assert self.robot_mg.go_to_pose_goal(self.CONTAINER_SCALE_POSE, wait=True)
        # Open gripper
        self.robot_mg.open_gripper(wait=True)
        # Retract
        assert self.robot_mg.go_to_pose_goal(offset_pose(self.CONTAINER_SCALE_POSE, [0.01, 0, 0]))
        assert self.robot_mg.go_to_pose_goal(offset_pose(self.CONTAINER_SCALE_POSE, [0.01, 0, 0.15]))
        assert self.robot_mg.go_to_joint_state(self.HOME, cartesian_path=True)

    def reset_containers(self):
        self.place_container_on_shelf(self.CONTAINER_SHELF_POSE_2, return_home=True)
        self.pick_container_from_scale()
        self.place_container_on_shelf(self.CONTAINER_SHELF_POSE_1)
        self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_2, return_home=True)
        self.place_container_on_scale()
        self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_1, return_home=True)

    def sample_weight(self) -> float:
        return np.random.uniform(self.MIN_DISPENSE_WEIGHT, min(self.MAX_DISPENSE_WEIGHT, self.available_weight))

    def interact(self, total_steps: int = None, policy: Optional[BasePolicy] = None) -> Dict:
        """
        Will dispense for the requested timesteps and returns the requested data
        """
        data = {}
        current_steps = 0

        while current_steps < total_steps:
            if self.available_weight < self.REFILL_THRESHOLD:
                rospy.loginfo(f"Container is empty. Reset sequence initiated.")
                self.reset_containers()
                continue

            # Proceed with dispensing
            target_wt = self.sample_weight()
            rospy.loginfo(f"[{current_steps}/{total_steps}]:\t Requested Wt: {target_wt:0.4f} g")
            success, dispensed_wt, rollout_data = self.dispenser.dispense_ingredient(
                ingredient_params=self.ingredient_params, target_wt=target_wt, policy=policy
            )
            self.available_weight -= max(0, dispensed_wt)
            rospy.loginfo(f"Available ingredient quantity: {self.available_weight:0.2f} g")

            if success:
                current_steps += len(rollout_data["obs"])

                for k, v in rollout_data.items():
                    if k in data:
                        data[k].append(v)
                    else:
                        data[k] = [v]

        for k, v in data.items():
            data[k] = np.concatenate(v, axis=0)

        return data

    @property
    def observation_space(self) -> gym.spaces.Space:
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        return observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        return action_space
