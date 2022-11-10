# Miscellaneous Info

[Presentation](https://docs.google.com/presentation/d/14sgyThLzdsNDLyY1ZwyCCyG6y1ywOnLFZ5BnoFrHwLM/edit?usp=sharing) of relevant stuff

## RL Formulation

Let $\omega$ be the angular velocity about the spatial axis about which the container is rotated and $e$ be the PID controller error.

The State Space is given as 

$ S = \begin{bmatrix} \omega_{t-1} & \dot{\omega}_{t-1} & \omega^{PID}_{t} & e_{t-1} & \dot{e}_{t-1} & e_{t-2} & \dot{e}_{t-2} & e_{t-3} & \dot{e}_{t-3} & e_{t-4} & \dot{e}_{t-4}  & e_{t-5} & \dot{e}_{t-5} \end{bmatrix} $

## 

```
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
pip install stable-baselines3
```


# Instructions


The following steps have to executed to run ingredient dispensing in isolation:

```
roslaunch ur_motion ur5e_bringup.launch robot_ip:=10.0.0.2
roslaunch learn2dispense setup.launch
roslaunch sensor_interface start_sensors.launch force_torque:=0 auto_cooking_station:=0 sensing_station:=0
roslaunch learn2dispense learn.launch
```

## Things to figure
- How to set gamma
- Clip high observations
- Clipping actions due to kinematic constraints
