# Miscellaneous Info

[Presentation](https://docs.google.com/presentation/d/14sgyThLzdsNDLyY1ZwyCCyG6y1ywOnLFZ5BnoFrHwLM/edit?usp=sharing) of relevant stuff

## RL Formulation

Let $\omega$ be the angular velocity about the spatial axis about which the container is rotated and $e$ be the PID controller error.

The State Space is given as 

$ S = \begin{bmatrix} \omega_{t-1} & \dot{\omega}_{t-1} & \omega^{PID}_{t} & e_{t-1} & \dot{e}_{t-1} & e_{t-2} & \dot{e}_{t-2} & e_{t-3} & \dot{e}_{t-3} & e_{t-4} & \dot{e}_{t-4}  & e_{t-5} & \dot{e}_{t-5} \end{bmatrix} $