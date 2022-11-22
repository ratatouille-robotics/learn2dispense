import numpy as np


def reward_func_v5(error: np.ndarray, error_rate: np.ndarray, success: bool) -> np.ndarray:
    """
    Second iteration of Reward function suggested by Oliver
    """
    e_penalty = (error / 200) ** 2
    e_dt_pentaly = (error_rate / 100) ** 2
    e_d2t_penalty = np.zeros_like(error_rate)
    e_d2t_penalty[1:] = error_rate[1:] - error_rate[:-1]
    rewards = -(e_penalty + e_dt_pentaly + e_d2t_penalty)
    if not success:
        rewards[-10:] *= 10

    return rewards


def reward_func_v4(error: np.ndarray, error_rate: np.ndarray, success: bool) -> np.ndarray:
    """
    Reward function used for run-v4
    """
    closeness_factor = np.maximum(0, 25 - error) / 25
    normalized_flow_rate = -np.maximum(np.minimum(0, error_rate), -50) / 50
    t_penalty = 0.1 * np.ones_like(error)
    q_penalty = closeness_factor * pow(normalized_flow_rate, 2)
    if not success:
        q_penalty[-20:] *= 10
        max_penalty_t = np.argmax(q_penalty[-10:])
        q_penalty[-10 + max_penalty_t:] = q_penalty[-10 + max_penalty_t]
    rewards = -(t_penalty + q_penalty)

    return rewards


def reward_func_v3(error: np.ndarray, error_rate: np.ndarray, success: bool) -> np.ndarray:
    """
    First iteration of Reward function suggested by Oliver
    """
    e_penalty = (error / 200) ** 2
    e_dt_pentaly = (error_rate / 200) ** 2
    rewards = -(e_penalty + e_dt_pentaly)
    if not success:
        rewards[-5:] *= 10

    return rewards
