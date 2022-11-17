import os
import tf
import gym
import yaml
import rospy
import torch
import pickle
import pathlib
import numpy as np

from geometry_msgs.msg import Pose
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from stable_baselines3.common.policies import BasePolicy

from motion.commander import RobotMoveGroup
from motion.utils import offset_pose, make_pose
from learn2dispense.dispense_rollout import Dispenser


class Environment:
    """
    Class to maintain an internal state of the cell and continuously run with running into errors.
    Generates data samples for learning by interacting with the environment.
    """
    HOME = [-0.5992, -2.4339, 2.2566, -2.9490, -1.0004, 3.0982]
    SHELF_FRONT = [-0.0142, -1.9426, 2.1134, -3.0728, 0.0594, 2.9016]
    CONTAINER_SHELF_POSE_1 = [-0.291, -0.472, 0.726, 0.729, -0.004, -0.008, 0.685]
    CONTAINER_SHELF_POSE_2 = [-0.291, -0.472, 0.477, 0.729, -0.004, -0.008, 0.685]
    CONTAINER_SCALE_POSE = [-0.436, 0.029, 0.214, -0.498, 0.521, 0.498, -0.482]
    MARKER_TO_TOOL0_OFFSET = [-0.0102, -0.0072, 0.127, 0.0587, 0.9962, -0.061, 0.0218]

    MIN_DISPENSE_WEIGHT = 10
    MAX_DISPENSE_WEIGHT = 100
    REFILL_THRESHOLD = 25
    ALMOST_CLOSED_STATE = 155
    EMPTY_CONTAINER_WEIGHT = 116.5
    TAG_ID = 15

    def __init__(
        self,
        log_dir: pathlib.Path,
        pick_container_on_start: bool = True,
        log_rollout: bool = False,
        available_weight: Optional[float] = None,
        use_fill_level: bool = False
    ) -> None:
        self.log_dir = log_dir
        self.log_rollout = log_rollout
        self.robot_mg = RobotMoveGroup()
        self.dispenser = Dispenser(robot_mg=self.robot_mg, use_fill_level=use_fill_level)
        self.tf_listener = tf.TransformListener()
        self._pre_process()
        self.num_episodes = 0
        self.num_batches = 0
        self.mode = "train"

        if self.log_rollout:
            if not os.path.exists(log_dir / "rollout_data"):
                os.makedirs(log_dir / "rollout_data")

        assert self.robot_mg.go_to_joint_state(self.HOME, cartesian_path=False)

        if pick_container_on_start:
            if available_weight is None:
                self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_1, return_home=True)
                self.place_container_on_scale()
                self.pick_container_from_scale()
                self.place_container_on_shelf(self.CONTAINER_SHELF_POSE_1)
            else:
                self.available_weight = available_weight
            # To start with, pick and place empty container on weighing scale
            self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_2, return_home=True)
            self.place_container_on_scale()
            # Pick full container for dispensing
            self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_1, return_home=True)

        # Load ingredient-specific params
        config_dir = pathlib.Path(__file__).parent.parent.parent
        with open(config_dir / "config/lentil_params.yaml") as f:
            self.ingredient_params = yaml.safe_load(f)

    def get_transformation_matrix(self, trans: List, quat: List) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        rot = R.from_quat(quat).as_matrix()
        matrix[:3, :3] = rot
        matrix[:3, -1] = trans

        return matrix

    def get_pose_from_matrix(self, matrix: np.ndarray) -> Pose:
        pos = matrix[:3, -1].tolist()
        quat = R.from_matrix(matrix[:3, :3]).as_quat().tolist()

        return make_pose(pos, quat)

    def set_mode(self, mode: str):
        assert mode in ["train", "test"]
        self.mode = mode

    def compute_pick_pose(self, prior_pose: Optional[Pose] = None) -> Pose:
        rospy.sleep(1)
        trans, quat = self.tf_listener.lookupTransform("base_link", f"ar_marker_{self.TAG_ID}", rospy.Time(0))
        base_T_marker = self.get_transformation_matrix(trans, quat)
        marker_T_tool0 = self.get_transformation_matrix(self.MARKER_TO_TOOL0_OFFSET[:3], self.MARKER_TO_TOOL0_OFFSET[3:])
        base_T_tool0 = base_T_marker @ marker_T_tool0
        new_pose = self.get_pose_from_matrix(base_T_tool0)

        if prior_pose is not None:
            new_pose.position.z = prior_pose.position.z
            new_pose.orientation = prior_pose.orientation

        return new_pose

    def _pre_process(self):
        self.CONTAINER_SCALE_POSE = make_pose(self.CONTAINER_SCALE_POSE[:3], self.CONTAINER_SCALE_POSE[3:])
        self.CONTAINER_SHELF_POSE_1 = make_pose(self.CONTAINER_SHELF_POSE_1[:3], self.CONTAINER_SHELF_POSE_1[3:])
        self.CONTAINER_SHELF_POSE_2 = make_pose(self.CONTAINER_SHELF_POSE_2[:3], self.CONTAINER_SHELF_POSE_2[3:])

    def smart_gripper_close(self, pose: Pose, scoop_offset: Optional[List] = None):
        assert self.robot_mg.close_gripper(wait=True, speed=10, force=5)
        assert self.robot_mg.open_gripper(wait=True, target_state=self.ALMOST_CLOSED_STATE, speed=10, force=5)
        scoop_pose = offset_pose(pose, scoop_offset)
        assert self.robot_mg.go_to_pose_goal(scoop_pose, acc_scaling=0.01, velocity_scaling=0.01)
        assert self.robot_mg.close_gripper(wait=True)
        assert self.robot_mg.go_to_pose_goal(offset_pose(scoop_pose, [0, 0, 0.015]))

    def smart_gripper_open(self):
        self.robot_mg.open_gripper(wait=True, force=5, speed=10)
        self.robot_mg.close_gripper(wait=True, force=5, speed=10)
        self.robot_mg.open_gripper(wait=True, force=5, speed=20)

    def pick_container_from_shelf(self, pick_pose, return_home: bool = False):
        # Open gripper
        self.robot_mg.open_gripper(wait=True)
        # Go to designated pose
        assert self.robot_mg.go_to_joint_state(self.SHELF_FRONT, cartesian_path=False)
        assert self.robot_mg.go_to_pose_goal(offset_pose(pick_pose, [0, 0.2, 0.0]), cartesian_path=False)
        # pick_pose = self.compute_pick_pose(pick_pose)
        # assert self.robot_mg.go_to_pose_goal(offset_pose(pick_pose, [0, 0.2, 0.0]), acc_scaling=0.1, velocity_scaling=0.1)
        assert self.robot_mg.go_to_pose_goal(pick_pose, wait=True)
        # Close gripper
        self.smart_gripper_close(pick_pose, scoop_offset=[0, -0.005, 0])
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
        self.smart_gripper_open()
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
        self.smart_gripper_open()
        # Retract
        assert self.robot_mg.go_to_pose_goal(offset_pose(self.CONTAINER_SCALE_POSE, [0.01, 0, 0]))
        assert self.robot_mg.go_to_pose_goal(offset_pose(self.CONTAINER_SCALE_POSE, [0.01, 0, 0.15]))
        assert self.robot_mg.go_to_joint_state(self.HOME, cartesian_path=True)

    def reset(self):
        rospy.loginfo("Resetting environment by emptying containers...")
        self.dispenser.run_reset_control(ingredient_params=self.ingredient_params)
        self.reset_containers()
        rospy.loginfo("Resetting complete...")

    def reset_containers(self):
        self.place_container_on_shelf(self.CONTAINER_SHELF_POSE_2, return_home=True)
        self.pick_container_from_scale()
        self.place_container_on_shelf(self.CONTAINER_SHELF_POSE_1)
        self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_2, return_home=True)
        self.place_container_on_scale()
        self.pick_container_from_shelf(self.CONTAINER_SHELF_POSE_1, return_home=True)

    def restore_initial_env_state(self):
        self.place_container_on_shelf(self.CONTAINER_SHELF_POSE_2, return_home=True)
        self.pick_container_from_scale()
        self.place_container_on_shelf(self.CONTAINER_SHELF_POSE_1)

    def sample_weight(self) -> float:
        return np.random.uniform(self.MIN_DISPENSE_WEIGHT, min(self.MAX_DISPENSE_WEIGHT, self.available_weight))

    def interact(
        self,
        total_steps: int = None,
        total_episodes: int = None,
        episode_list: Optional[List[float]] = None,
        policy: Optional[BasePolicy] = None,
        eval_mode: bool = False
    ) -> Tuple[Dict, List]:
        """
        Will dispense for the requested timesteps and returns the requested data
        """
        assert total_steps is not None or total_episodes is not None or episode_list is not None
        assert total_steps is None or total_episodes is None
        data = {}
        infos = []
        current_step = 0
        current_episode = 0
        self.num_batches += 1

        if episode_list is not None:
            total_episodes = len(episode_list)

        while (total_steps is None or current_step < total_steps) and (
            total_episodes is None or current_episode < total_episodes
        ):
            if (episode_list is not None and self.available_weight < episode_list[current_episode]
            ) or self.available_weight < self.REFILL_THRESHOLD:
                rospy.loginfo(f"Container is empty. Reset sequence initiated.")
                self.reset_containers()
                continue

            # Decide how much weight to request for current episode
            if episode_list is None:
                target_wt = self.sample_weight()
            else:
                target_wt = episode_list[current_episode]

            if total_episodes is None:
                rospy.loginfo(f"[{current_step}/{total_steps}]:\t Requested Wt: {target_wt:0.4f} g")
            else:
                rospy.loginfo(f"[{current_episode}/{total_episodes}]:\t Requested Wt: {target_wt:0.4f} g")
            
            completed, dispensed_wt, rollout_data, info = self.dispenser.dispense_ingredient(
                ingredient_params=self.ingredient_params,
                target_wt=target_wt,
                policy=policy,
                eval_mode=eval_mode,
                ingredient_wt_start=self.available_weight
            )
            self.available_weight -= max(0, dispensed_wt)
            rospy.loginfo(f"Available ingredient quantity: {self.available_weight:0.2f} g")

            if completed:
                self.num_episodes += 1
                current_step += len(rollout_data["obs"])
                current_episode += 1
                infos.append(info)

                if self.mode == "train":
                    for k, v in rollout_data.items():
                        if k in data:
                            data[k].append(v)
                        else:
                            data[k] = [v]

        if self.mode == "train":
            for k, v in data.items():
                if (isinstance(v[0], np.ndarray)):
                    data[k] = np.concatenate(v, axis=0)
                else:
                    data[k] = torch.cat(v, dim=0)

        if self.log_rollout:
            data_copy = {}
            for k, v in data.items():
                data_copy[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            
            with open(self.log_dir / "rollout_data" / f"batch_{self.num_batches}", "wb") as f:
                pickle.dump(data_copy, f)

        return data, infos

    @property
    def observation_space(self) -> gym.spaces.Space:
        obs_size = np.sum(self.dispenser.OBS_HIST_LENGTH)
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        return observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        return action_space
