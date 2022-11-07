#!/usr/bin/env python3
import sys
import rospy
import pathlib

from learn2dispense.env import Environment


if  __name__ == "__main__":
    try:
        rospy.init_node("learn_dispense")
        log_dir = pathlib.Path(__file__).parent.parent / "logs"

        env = Environment(log_dir=log_dir)
        env.interact(10000)

    except rospy.ROSInterruptException:
        sys.exit(1)
