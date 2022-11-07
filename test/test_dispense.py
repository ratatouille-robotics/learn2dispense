#!/usr/bin/env python3
import os
import sys
import csv
import yaml
import time
import rospy
import pathlib
import numpy as np

from datetime import datetime

from motion.commander import RobotMoveGroup
from motion.utils import make_pose
from learn2dispense.dispense_rollout import Dispenser
from learn2dispense.env import HOME


LOG_DIR = "src/learn2dispense/logs"


def acquire_input(message: str) -> float:
    """
    Get weight to be dispensed from the user
    """
    input_wt = input(message)

    try:
        input_wt = float(input_wt)
    except ValueError:
        input_wt = -1

    return input_wt


def run(log_results=False):
    rospy.init_node("ur5e_dispense_test")
    robot_mg = RobotMoveGroup()
    dispenser = Dispenser(robot_mg)
    num_runs = 0

    if log_results:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        log_file = "eval_{0}.csv".format(datetime.now().strftime("%b-%d--%H-%M-%S"))
        out_file = open(LOG_DIR + "/" + log_file, "w")
        csv_writer = csv.writer(out_file)
        csv_writer.writerow(["S.No", "Requested", "Dispensed", "Time Taken"])

    input_wt = acquire_input("Enter desired ingredient quantity (in grams): ")

    while (input_wt) > 0 and float(input_wt) <= 1000:
        num_runs += 1
        # Move to dispense-home position
        assert robot_mg.go_to_joint_state(HOME, cartesian_path=True)

        # Load ingredient-specific params
        config_dir = pathlib.Path(__file__).parent.parent
        with open(config_dir / "config/lentil_params.yaml") as f:
            params = yaml.safe_load(f)

        # Dispense ingredient
        start_time = time.time()
        _, dispensed_wt, _ = dispenser.dispense_ingredient(params, float(input_wt))
        dispense_time = time.time() - start_time
        print(f"Dispensing Time: {dispense_time: 0.2f}s")

        if log_results:
            csv_writer.writerow([num_runs, input_wt, np.round(dispensed_wt, 2), np.round(dispense_time, 1)])
            out_file.flush()

        # Get next entry from user
        input_wt = acquire_input("Enter desired ingredient quantity (in grams): ")
    
    if log_results:
        out_file.close()


if __name__ == "__main__":
    try:
        run(log_results=False)
    except rospy.ROSInterruptException:
        sys.exit(1)
