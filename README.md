# Instructions

The following packages are necessary to run the ingredient dispensing:

- [ur_motion](https://github.com/ratatouille-robotics/ur_motion)

- [sensor_interface](https://github.com/ratatouille-robotics/sensor_interface)

- [dispense](https://github.com/ratatouille-robotics/dispense)
<BR> <BR>

The following steps have to executed to run ingredient dispensing in isolation:

1. Turn on the robot. Connect the robot and control PC using LAN. Turn on the weighing scale and connect it to the control PC.

2. On the ROS side, start the Universal Robot ROS Driver to interface with the robot

    ```
    roslaunch ur_motion ur5e_bringup.launch robot_ip:=10.0.0.2
    ```

3. Start the MoveIt Motion Planning interface
    ```
    roslaunch learn2dispense setup.launch
    ```

4. Start the streaming of weighing scale measurements by running
    ```
    roslaunch sensor_interface start_sensors.launch force_torque:=0 auto_cooking_station:=0 sensing_station:=0
    ```
5. Load the correct program on the UR pedant. Run the program so that the robot can
receive commands from the control PC.

6. Run the following command to begin the dispensing sequence
    ```
    roslaunch learn2dispense learn.launch
    ```