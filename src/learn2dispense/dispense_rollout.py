#!/usr/bin/env python3
import time
import rospy
import numpy as np
import torch as th
from typing import Tuple, Dict, Optional
from collections import deque

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import obs_as_tensor
from motion.utils import make_pose
from motion.commander import RobotMoveGroup
from sensor_interface.msg import Weight
import dispense.transforms as T
from dispense.dispense import get_transform, get_rotation


T_STEP = 0.1
CONTROL_STEP = 0.005

MAX_ROT_ACC = np.pi / 4
MIN_ROT_ACC = -2 * MAX_ROT_ACC
MAX_ROT_VEL = np.pi / 32
MIN_ROT_VEL = -2 * MAX_ROT_VEL

LEARNING_MAX_VEL = MAX_ROT_VEL / 2

ANGLE_LIMIT = {
    "regular": {
        "corner": (1 / 3) * np.pi,
        "edge": (1 / 2) * np.pi
    },
    "spout": {"corner": (2 / 5) * np.pi}
}

DERIVATIVE_WINDOW = 0.1  # Time window over which to calculate derivative. Has to be greater than equal to T_STEP

# Offsets from wrist_link_3/flange/tool0
CONTAINER_OFFSET = {
    "regular": np.array([0.040, 0.060, 0.250], dtype=np.float),
    "spout": np.array([0.035, 0.150, 0.250], dtype=np.float),
    "holes": np.array([0.040, 0.060, 0.250], dtype=np.float),
}

POURING_POSES = {
    "regular": {
        "corner": ([-0.375, -0.020, 0.486], [0.671, -0.613, -0.414, 0.048]),
        "edge": ([-0.425, 0.245, 0.520], [0.910, -0.324, -0.109, 0.235]),
    },
    "spout": {"corner": ([-0.295, -0.03, 0.460], [0.633, -0.645, -0.421, 0.082])},
    "holes": {"corner": ([-0.360, 0.070, 0.520], [0.749, 0.342, -0.520, -0.228])},
}


class Dispenser:

    OBS_DATA = ["error", "error_rate", "velocity", "acceleration", "pid_output"]
    OBS_HIST_LENGTH = [5, 5, 1, 1, 1]
    OBS_MEAN = [50, -25, 0, 0, 0]
    OBS_STD = [50, 25, MAX_ROT_VEL, MAX_ROT_ACC, MAX_ROT_VEL]

    def __init__(self, robot_mg: RobotMoveGroup) -> None:
        assert (T_STEP / CONTROL_STEP).is_integer()
        # Setup comm with the weighing scale
        self.wt_subscriber = rospy.Subscriber("/cooking_pot/weighing_scale", Weight, callback=self._weight_callback)
        self.rate = rospy.Rate(1 / CONTROL_STEP)
        self.robot_mg = robot_mg
        self._w_data = None

    def _weight_callback(self, data: float) -> None:
        self._w_data = data

    def get_weight(self) -> float:
        if self._w_data is None:
            rospy.logerr("No values received from the publisher")
            raise

        return self._w_data.weight

    def get_weight_fb(self) -> Tuple[float, bool]:
        return (
            self.get_weight(),
            (rospy.Time.now() - self._w_data.header.stamp).to_sec() < 0.5,
        )

    def get_last_state(self) -> np.ndarray:
        last_obs = []
        for i, obs_item in enumerate(self.OBS_DATA):
            avail_size = len(self.rollout_data[obs_item])
            reqd_size = self.OBS_HIST_LENGTH[i]
            if avail_size >= reqd_size:
                last_obs.append(self.rollout_data[obs_item][-reqd_size:])
            else:
                last_obs.append(self.rollout_data[obs_item][-avail_size:])
                last_obs.append((reqd_size - avail_size) * [self.rollout_data[obs_item][0],])

        last_obs = np.concatenate(last_obs, axis=-1)

        return last_obs

    def reset_rollout(self):
        self.rollout_data = {}
        for k in self.OBS_DATA:
            self.rollout_data[k] = []

    def process_rollout_data(self) -> Dict:
        outputs = {}
        for k, v in self.rollout_data.items():
            if isinstance(v, th.Tensor):
                self.rollout_data[k] = th.tensor(v, dtype=th.float32)
            else:
                self.rollout_data[k] = np.array(v, dtype=np.float32)

        obs = []
        for i, obs_item in enumerate(self.OBS_DATA):
            data = np.zeros((self.steps, self.OBS_HIST_LENGTH[i]), dtype=np.float32)
            for t_step in range(self.OBS_HIST_LENGTH[i]):
                data[t_step:, t_step] = self.rollout_data[obs_item][: self.steps - t_step]
                if obs_item in ["error"]:
                    data[:t_step, t_step] = data[0, 0]
            data = (data - self.OBS_MEAN[i]) / self.OBS_STD[i]
            obs.append(data)

        obs = np.concatenate(obs, axis=1)
        outputs["obs"] = obs

        outputs["episode_start"] = np.zeros(len(outputs["obs"]), dtype=np.float32)
        outputs["episode_start"][0] = 1
        outputs["time"] = self.rollout_data["time"]

        return outputs

    def dispense_ingredient(
        self,
        ingredient_params: dict,
        target_wt: float,
        policy: Optional[BasePolicy] = None
    ) -> Tuple[bool, float, Dict]:
        # Record current robot position
        robot_original_pose = self.robot_mg.get_current_pose()
        # Send dummy velocity to avoid delayed motion start on first run
        self.robot_mg.send_cartesian_vel_trajectory(T.numpy2twist(np.zeros(6, dtype=np.float)))

        # set ingredient-specific params
        self.ctrl_params = ingredient_params["controller"]
        self.lid_type = ingredient_params["container"]["lid"]
        if self.lid_type in ["none", "slot"]:
            self.lid_type = "regular"
        self.container_offset = CONTAINER_OFFSET[self.lid_type]

        # set ingredient-specific limits
        self.max_rot_vel = self.ctrl_params["vel_scaling"] * MAX_ROT_VEL
        self.min_rot_vel = self.ctrl_params["vel_scaling"] * MIN_ROT_VEL
        self.max_rot_acc = self.ctrl_params["acc_scaling"] * MAX_ROT_ACC
        self.min_rot_acc = self.ctrl_params["vel_scaling"] * MIN_ROT_ACC

        # Move to dispense-start position
        pos, orient = POURING_POSES[self.lid_type][ingredient_params["pouring_position"]]
        pre_dispense_pose = make_pose(pos, orient)
        assert self.robot_mg.go_to_pose_goal(
            pre_dispense_pose, cartesian_path=True, orient_tolerance=0.05, velocity_scaling=0.75, acc_scaling=0.5
        )

        # set run-specific params
        self.rate.sleep()
        self.reset_rollout()
        self.start_wt = self.get_weight()
        self.last_vel = 0
        self.last_acc = 0

        # Dispense ingredient
        rospy.loginfo("Dispensing started...")
        self.angle_limit = ANGLE_LIMIT[self.lid_type][ingredient_params["pouring_position"]]
        self.run_pd_control(target_wt, self.ctrl_params["error_threshold"], policy)

        # Move to dispense-start position
        assert self.robot_mg.go_to_pose_goal(
            pre_dispense_pose, cartesian_path=True, orient_tolerance=0.05, velocity_scaling=0.5, acc_scaling=0.25
        )

        assert self.robot_mg.go_to_pose_goal(
            robot_original_pose, cartesian_path=True, orient_tolerance=0.05, velocity_scaling=0.75, acc_scaling=0.5
        )

        success = False
        dispensed_wt = self.get_weight() - self.start_wt
        if (target_wt - dispensed_wt) > ingredient_params["tolerance"]:
            rospy.logerr(f"Dispensed amount is below tolerance...")
            rospy.logerr(f"Dispensed Wt: {dispensed_wt:0.2f}g")
        elif (dispensed_wt - target_wt) > ingredient_params["tolerance"]:
            rospy.logerr(f"Dispensed amount exceeded the tolerance...")
            rospy.logerr(f"Dispensed Wt: {dispensed_wt:0.2f}g")
        else:
            rospy.loginfo(f"Ingredient dispensed successfuly...")
            rospy.loginfo(f"Dispensed Wt: {dispensed_wt:0.2f} g")
            success = True

        return success, dispensed_wt, self.process_rollout_data()

    def run_control_loop(self, velocity):
        last_velocity = self.last_vel
        for _ in range(int(T_STEP / CONTROL_STEP)):
            if velocity > last_velocity:
                curr_velocity = min(last_velocity + self.max_rot_acc * CONTROL_STEP, velocity)
            else:
                curr_velocity = max(last_velocity + self.min_rot_acc * CONTROL_STEP, velocity)

            # Convert the velocity into a twist
            raw_twist = curr_velocity * self.base_raw_twist

            # Transform the frame of the twist
            curr_pose = self.robot_mg.get_current_pose()
            twist_transform = get_transform(curr_pose, self.container_offset)
            twist = T.TransformTwist(raw_twist, twist_transform)
            twist = T.numpy2twist(twist)

            self.robot_mg.send_cartesian_vel_trajectory(twist)
            self.rate.sleep()
            last_velocity = curr_velocity

    def run_pd_control(self, target_wt: float, err_threshold: float, policy: BasePolicy):
        """
        Run the PD controller
        """
        assert DERIVATIVE_WINDOW >= T_STEP
        start_T = T.pose2matrix(self.robot_mg.get_current_pose())

        error = target_wt
        wt_fb_acc = deque(maxlen=int(DERIVATIVE_WINDOW / T_STEP) + 1)
        self.base_raw_twist = np.array([0, 0, 0] + self.ctrl_params["rot_axis"], dtype=np.float)
        success = True
        start_time = time.time()
        self.steps = 0

        # Run controller as long as error is not within tolerance
        while error > err_threshold:
            iter_start_time = time.time() - start_time
            curr_wt, is_recent = self.get_weight_fb()
            wt_fb_acc.append(curr_wt)
            if not is_recent:
                rospy.logerr("Weight feedback from weighing scale is too delayed. Stopping dispensing process.")
                success = False
                break

            error = target_wt - (curr_wt - self.start_wt)
            mean_curr_wt = wt_fb_acc[-1]
            mean_prev_wt = wt_fb_acc[0]
            error_rate = -(mean_curr_wt - mean_prev_wt) / DERIVATIVE_WINDOW

            p_term = self.ctrl_params["p_gain"] * error
            p_term = min(p_term, self.max_rot_vel)  # clamp p-term
            d_term = self.ctrl_params["d_gain"] * error_rate
            pid_vel = p_term + d_term

            # Clamp velocity based on acceleration and velocity limits
            delta_vel = pid_vel - self.last_vel
            if np.sign(delta_vel) == 1 and delta_vel / T_STEP > self.max_rot_acc:
                pid_vel = self.last_vel + self.max_rot_acc * T_STEP
            elif np.sign(delta_vel) == -1 and delta_vel / T_STEP < self.min_rot_acc:
                pid_vel = self.last_vel + self.min_rot_acc * T_STEP
            pid_vel = np.clip(pid_vel, self.min_rot_vel, self.max_rot_vel)

            total_vel = pid_vel

            if policy is not None:
                with th.no_grad():
                    obs = self.get_last_state()
                    obs_tensor = obs_as_tensor(obs, policy.device)
                    action, value, log_prob = policy(obs_tensor)
                action = action.cpu().numpy()

                total_vel += (LEARNING_MAX_VEL * action)

            self.run_control_loop(total_vel)

            self.rollout_data["time"].append(iter_start_time)
            self.rollout_data["velocity"].append(self.last_vel)
            self.rollout_data["acceleration"].append(self.last_acc)
            self.rollout_data["pid_output"].append(pid_vel)
            self.rollout_data["error"].append(error)
            self.rollout_data["error_rate"].append(error_rate)
            self.rollout_data["action"].append(action)
            self.rollout_data["value"].append(value)
            self.rollout_data["log_prob"].append(log_prob)
            self.steps += 1

            self.last_acc = (total_vel - self.last_vel) / T_STEP
            self.last_vel = total_vel

            # Check if the angluar limits about the pouring axis have been reached
            curr_pose = self.robot_mg.get_current_pose()
            if np.abs(get_rotation(start_T, T.pose2matrix(curr_pose))[0]) >= self.angle_limit:
                rospy.logerr("Container does not seem to have sufficient ingredient quantity...")
                success = False
                break

        self.min_rot_vel = min(2 * MIN_ROT_VEL, self.min_rot_vel)
        # Retract the container to the starting position
        while True:
            curr_wt = self.get_weight()

            curr_pose = self.robot_mg.get_current_pose()
            ang, ax = get_rotation(start_T, T.pose2matrix(curr_pose))
            ang = np.sign(np.dot(ax, self.ctrl_params["rot_axis"])) * ang

            if ang < 0.005:
                break

            # Ensure the velocity is within limits
            vel = -2 * ang
            delta_vel = vel - self.last_vel
            if np.sign(delta_vel) == 1 and delta_vel / T_STEP > self.max_rot_acc:
                vel = self.last_vel + self.max_rot_acc * T_STEP
            elif np.sign(delta_vel) == -1 and delta_vel / T_STEP < self.min_rot_acc:
                vel = self.last_vel + self.min_rot_acc * T_STEP
            vel = np.clip(vel, self.min_rot_vel, self.max_rot_vel)

            self.run_control_loop(vel)
            self.last_acc = (vel - self.last_vel) / T_STEP
            self.last_vel = vel

        return success
