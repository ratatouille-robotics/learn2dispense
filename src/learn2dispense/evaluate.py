#!/usr/bin/env python3
import sys
import rospy
import pathlib

from learn2dispense.env import Environment
from learn2dispense.ppo import PPO


RUN_NAME = "learn_dispense"


LOG_DIR = pathlib.Path(__file__).parent.parent.parent / f"eval_logs/{RUN_NAME}"
MODEL_PATH = pathlib.Path(__file__).parent.parent.parent / "logs/learn_dispense_v4/model/best_iter_27"
AVAILABLE_WT = 1097.5


if __name__ == "__main__":
    try:
        rospy.init_node("evaluate_dispense")

        env = Environment(
            log_dir=LOG_DIR,
            log_rollout=True,
            use_fill_level=True,
            available_weight=AVAILABLE_WT
        )

        model = PPO.load(
            path=MODEL_PATH,
            env=env
        )

        data, info = env.interact(
            episode_list=[64, 58, 71, 30, 90, 40, 21, 41, 43, 76, 66, 98, 73, 62, 69, 37, 19, 55, 22, 45, 49, 81, 45, 29, 66],
            policy=model.policy,
            eval_mode=True
        )

        env.restore_initial_env_state()

    except rospy.ROSInterruptException:
        sys.exit(1)
