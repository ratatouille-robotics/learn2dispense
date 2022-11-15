#!/usr/bin/env python3
import sys
import rospy
import pathlib
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR

from learn2dispense.env import Environment
from learn2dispense.policy import SimplePolicy
from learn2dispense.ppo import PPO


def lambda_schedule(epoch: int) -> float:
    if epoch < 1e5:
        return 1
    elif epoch < 2e5:
        return 0.25
    else:
        return 0.1


if __name__ == "__main__":
    try:
        rospy.init_node("learn_dispense")
        exp_tag = datetime.now().strftime("%b-%d--%H-%M-%S")
        log_dir = pathlib.Path(__file__).parent.parent.parent / "logs" / "run_{0}".format(exp_tag)

        env = Environment(log_dir=log_dir, log_rollout=True, use_fill_level=True)

        model = PPO(
            policy=SimplePolicy,
            env=env,
            log_dir=log_dir,
            batch_size=256,
            n_steps=8192,
            checkpoint_freq=20000,
            lr_scheduler=LambdaLR,
            scheduler_kwargs={"lr_lambda": lambda_schedule}
        )
        model.learn(total_timesteps=int(1e6))
        env.restore_initial_env_state()

    except rospy.ROSInterruptException:
        sys.exit(1)
