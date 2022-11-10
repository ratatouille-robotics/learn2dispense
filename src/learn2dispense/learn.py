#!/usr/bin/env python3
import sys
import rospy
import pathlib
from datetime import datetime

from learn2dispense.env import Environment
from learn2dispense.policy import SimplePolicy
from learn2dispense.ppo import PPO


if __name__ == "__main__":
    try:
        rospy.init_node("learn_dispense")
        exp_tag = datetime.now().strftime("%b-%d--%H-%M-%S")
        log_dir = pathlib.Path(__file__).parent.parent.parent / "logs" / "run_{0}".format(exp_tag)

        env = Environment(log_dir=log_dir, log_rollout=True)

        model = PPO(
            policy=SimplePolicy,
            env=env,
            log_dir=log_dir
        )
        model.learn(total_timesteps=100000)

    except rospy.ROSInterruptException:
        sys.exit(1)
