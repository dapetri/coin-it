import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from models.sac.sac import SoftActorCritic
from visualization.visualize import ProgressBar, evaluate, v_function_visualization, plot_metrics,


class SACArgs:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    num_training_steps = 51000  # @param {type: "integer"}
    learning_rate = 3.0e-4  # @param {type: "number"}
    entropy_alpha = 0.02  # @param {type: "number"}
    discount_factor = 0.98  # @param {type: "number"}
    batch_size = 64  # @param {type: "integer"}
    buffer_limit = 100000  # @param {type: "integer"}
    reward_scale = 0.1  # @param {type: "number"}
    q_net_update_rate = 0.002  # @param {type: "number"}


class SACTrainer:
    def __init__(
        self,
        args: SACArgs,
    ) -> None:
        self.args = args

    def train(self):
        # import data
        environment = gym.make('Pendulum-v0')
        evaluation_environment = DummyVecEnv([lambda: gym.make('Pendulum-v0')])
        # keep a second environment for evaluation purposes.
        # We wrap it in a Dummy Vector Environment for compatibility with
        # the visualization utility

        soft_actor_critic = SoftActorCritic(args=self.args)

        reward_scale = self.args["reward_scale"]
        num_training_steps = self.args["num_training_steps"]

        # logging utility
        logging_frequency = 100  # log progress every 100 steps
        plot_frequency = 5000
        progress_bar = ProgressBar(num_iterations=num_training_steps)

        # restart the environment, i.e., go back to some initial state
        state = environment.reset()
        full_metrics = {"score": []}
        train_step_metrics = {}
        for current_step in range(num_training_steps):
            if current_step % logging_frequency == 0:  # log every logging_frequency steps
                for key, value in train_step_metrics.items():
                    if key not in full_metrics:
                        full_metrics[key] = []
                    full_metrics[key].append(value)

                evaluation_recordings = evaluate(evaluation_environment=evaluation_environment,
                                                 soft_actor_critic=soft_actor_critic)

                progress_bar(_steps=logging_frequency,
                             score=evaluation_recordings.get("score"),
                             **train_step_metrics)

                full_metrics["score"].append(
                    evaluation_recordings.get("score"))

            if current_step % plot_frequency == 0:  # plot visualizations
                v_function_visualization(evaluation_environment=evaluation_environment,
                                         soft_actor_critic=soft_actor_critic,
                                         current_step=current_step)
                visualize_rollout(soft_actor_critic=soft_actor_critic,
                                  step=current_step)
                plot_metrics(full_metrics)

            # alternate between collecting one step of data and updating SAC with one mini-batch
            action, log_probabilities = soft_actor_critic.policy(
                torch.from_numpy(np.array(state)).float())
            next_state, reward, done, info = environment.step(
                [environment.action_space.high[0] * action.item()])

            # safe (s, a, r, s', done) tuple in memory buffer
            soft_actor_critic.memory.put(
                (state, action.item(), reward * reward_scale, next_state, done))

            if done:
                state = environment.reset()
            else:
                state = next_state

            # wait until there are enough rollouts in the memory buffer before starting the training
            if soft_actor_critic.memory.size() > 1000:
                train_step_metrics = soft_actor_critic.train_step()
                train_step_metrics["buffer_size"] = soft_actor_critic.memory.size(
                )
        environment.close()


def main():
    pass


if __name__ == '__main__':
    main()
