#!/usr/bin/env python3
import tf
import sys
import rospy
import numpy as np

from scipy.spatial.transform import Rotation as R

from motion.commander import RobotMoveGroup
from motion.utils import make_pose, offset_pose


SHELF_FRONT = [-0.0142, -1.9426, 2.1134, -3.0728, 0.0594, 2.9016]
CONTAINER_SHELF_POSE_1 = [-0.291, -0.472, 0.726, 0.729, -0.004, -0.008, 0.685]
CONTAINER_SHELF_POSE_2 = [-0.291, -0.472, 0.477, 0.729, -0.004, -0.008, 0.685]


def get_T_matrix(pos, quat):
    rot = R.from_quat(quat).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, -1] = pos
    T[:3, :3] = rot
    return T


def get_offset(T):
    pos = T[:3, -1]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    print(pos, "\t", quat)
    return pos, quat


def run():
    rospy.init_node("calibrate")
    robot_mg = RobotMoveGroup()
    np.set_printoptions(precision=4)

    pose_1 = make_pose(CONTAINER_SHELF_POSE_1[:3], CONTAINER_SHELF_POSE_1[3:])
    pose_2 = make_pose(CONTAINER_SHELF_POSE_2[:3], CONTAINER_SHELF_POSE_2[3:])

    TRANS = []
    QUAT = []

    for i in range(3):
        _ = input("Please reset container and press enter")
        assert robot_mg.go_to_joint_state(SHELF_FRONT, cartesian_path=False)
        assert robot_mg.go_to_pose_goal(offset_pose(pose_1, [0, 0.2, 0.0]), cartesian_path=False)
        assert robot_mg.go_to_pose_goal(pose_1, wait=True)
        assert robot_mg.close_gripper(wait=True, force=5)
        assert robot_mg.open_gripper(wait=True, force=5)
        assert robot_mg.go_to_pose_goal(offset_pose(pose_1, [0, 0.2, 0.0]))
        rospy.sleep(2)

        tf_listener = tf.TransformListener()
        rospy.sleep(1)
        trans_1, quat_1 = tf_listener.lookupTransform('base_link', 'ar_marker_15', rospy.Time(0))
        T_1 = get_T_matrix(trans_1, quat_1)
        P_1 = get_T_matrix(CONTAINER_SHELF_POSE_1[:3], CONTAINER_SHELF_POSE_1[3:])
        offset_1 = get_offset(np.linalg.inv(T_1) @ P_1)
        TRANS.append(offset_1[0])
        QUAT.append(offset_1[1])

        assert robot_mg.go_to_pose_goal(offset_pose(pose_2, [0, 0.2, 0.0]), cartesian_path=False)
        assert robot_mg.go_to_pose_goal(pose_2, wait=True)
        assert robot_mg.close_gripper(wait=True, force=5)
        assert robot_mg.open_gripper(wait=True, force=5)
        assert robot_mg.go_to_pose_goal(offset_pose(pose_2, [0, 0.2, 0.0]))
        rospy.sleep(2)

        trans_2, quat_2 = tf_listener.lookupTransform('base_link', 'ar_marker_15', rospy.Time(0))
        T_2 = get_T_matrix(trans_2, quat_2)
        P_2 = get_T_matrix(CONTAINER_SHELF_POSE_2[:3], CONTAINER_SHELF_POSE_2[3:])
        offset_2 = get_offset(np.linalg.inv(T_2) @ P_2)
        TRANS.append(offset_2[0])
        QUAT.append(offset_2[1])
        print()

    print("Translation: ", np.mean(TRANS, axis=0))
    print("Rotation: ", np.mean(QUAT, axis=0))
    


if __name__ == "__main__":
    try:
        run()
    except rospy.ROSInterruptException:
        sys.exit(1)
