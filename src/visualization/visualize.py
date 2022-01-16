import os
import gym
import copy
import tqdm
import time
import torch
import numpy as np
import torch.nn as nn
import collections
import random
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from torch.distributions import Normal

data_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data', 'results')


# # disable the actual display to prevent errors with colab
# from pyvirtualdisplay import Display
# _display = Display(visible=False, size=(1400, 900))
# _ = _display.start()

class ProgressBar:
    def __init__(self, num_iterations: int, verbose: bool = True):
        if verbose:  # create a nice little progress bar
            self.scalar_tracker = tqdm.tqdm(total=num_iterations, desc="Scalars", bar_format="{desc}",
                                            position=0, leave=True)
            progress_bar_format = '{desc} {n_fmt:' + str(
                len(str(num_iterations))) + '}/{total_fmt}|{bar}|{elapsed}<{remaining}'
            self.progress_bar = tqdm.tqdm(total=num_iterations, desc='Iteration', bar_format=progress_bar_format,
                                          position=1, leave=True)
        else:
            self.scalar_tracker = None
            self.progress_bar = None

    def __call__(self, _steps: int = 1, **kwargs):
        if self.progress_bar is not None:
            formatted_scalars = {key: "{:.3e}".format(value[-1] if isinstance(value, list) else value)
                                 for key, value in kwargs.items()}
            description = ("Scalars: " + "".join([str(key) + "=" + value + ", "
                                                  for key, value in formatted_scalars.items()]))[:-2]
            self.scalar_tracker.set_description(description)
            self.progress_bar.update(_steps)

# this function will automatically save your figure into data/results (if correctly mounted!)


def save_figure(save_name: str) -> None:
    assert save_name is not None, "Need to provide a filename to save to"
    plt.savefig(os.path.join(data_path, save_name + ".png"))


def evaluate_rollout(evaluation_environment: gym.Env, soft_actor_critic) -> float:
    """
    Performs a full rollout using the mean of the current policy.
    :param evaluation_environment: The environment used for evaluation. In our case, a Pendulum environment
    :param soft_actor_critic: An instance of the SAC class defined above
    :return: The total score for the rollout
    """
    done = False
    score = 0
    state = evaluation_environment.reset()
    while not done:  # alternate between collecting one step of data and updating SAC with one mini-batch
        action, log_probabilities = soft_actor_critic.policy(
            torch.from_numpy(np.array(state)).float(), deterministic=True)
        scaled_action = evaluation_environment.action_space.high[0] * action.item(
        )
        next_state, reward, done, info = evaluation_environment.step([
                                                                     [scaled_action]])
        # need to wrap action in a list because of the video recording
        state = next_state  # go to the next environment step
        score += reward  # keep track of cumulative reward for recording
    return score


def v_function_visualization(evaluation_environment: gym.Env,
                             soft_actor_critic,
                             current_step: int = 0,
                             resolution: int = 100):
    """
    Visualizes a numerical approximation of the value function by evaluating the Q-Function for a wide range
    :param evaluation_environment:
    :param soft_actor_critic:
    :param current_step:
    :param resolution:
    :return:
    """
    plt.clf()

    max_speed = 8
    x = np.linspace(-np.pi, np.pi, num=resolution)
    y = np.linspace(-max_speed, max_speed, num=resolution)
    state_evaluation_grid = np.transpose(
        [np.tile(x, len(y)), np.repeat(y, len(x))])
    input_observations = torch.Tensor(np.array([np.cos(state_evaluation_grid[:, 0]),
                                                np.sin(
                                                    state_evaluation_grid[:, 0]),
                                                state_evaluation_grid[:, 1]])).T

    evaluations = []

    for position, action in enumerate(np.linspace(evaluation_environment.action_space.low,
                                                  evaluation_environment.action_space.high,
                                                  50)):
        action_tensor = torch.Tensor(
            np.full((len(state_evaluation_grid), 1), fill_value=action))

        q1_val = soft_actor_critic.q_net_1(input_observations, action_tensor)
        q2_val = soft_actor_critic.q_net_2(input_observations, action_tensor)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        reward_evaluation_grid = torch.min(q1_q2, 1, keepdim=True)[0]
        reward_evaluation_grid = reward_evaluation_grid.reshape(
            (resolution, resolution))

        evaluations.append(reward_evaluation_grid.detach().numpy())

    plt.title(f"Numerically integrated V-function at step {current_step}")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    heatmap = plt.contourf(x, y, np.array(evaluations).max(axis=0), levels=100,
                           cmap=plt.get_cmap("jet"), zorder=0)
    plt.colorbar(heatmap)
    save_figure(save_name=f"numerical_v_function_{current_step:04d}")


def plot_metrics(metrics: Dict[str, List[float]]):
    """
    Plots various metrics recorded during training
    :param metrics:
    :return:
    """
    if len(metrics) > 0:
        plt.clf()
        plt.figure(figsize=(16, 9))
        for position, (key, value) in enumerate(metrics.items()):
            plt.subplot(len(metrics), 1, position + 1)
            plt.plot(range(len(value)), np.array(value))
            plt.ylabel(key.title())
        plt.xlabel("Recorded Steps")
        plt.tight_layout()
        save_figure(f"training_metrics")
        plt.clf()
        plt.close()


def evaluate(evaluation_environment: gym.Env, soft_actor_critic,
             num_evaluation_rollouts: int = 10):
    """
    Perform num_evaluation_rollouts rollouts on the evaluation environment using the current policy and average over
    the achieved scores. Also plot a visualization of the first of these rollouts and a numerical integration
    of the value function
    :param evaluation_environment: The environment to evaluate. Will perform num_evaluation_rollouts full rollouts on
      this environment
    :param soft_actor_critic: Instance of SAC used to determine the actions
    :param num_evaluation_rollouts: Number of rollouts to evaluate for
    :return:
    """
    scores = []
    for rollout_idx in range(num_evaluation_rollouts):
        rollout_score = evaluate_rollout(evaluation_environment=evaluation_environment,
                                         soft_actor_critic=soft_actor_critic
                                         )
        scores.append(rollout_score)
    mean_score = np.mean(scores)
    return {"score": mean_score}
