import copy
import torch
from typing import Dict

from models.sac.q_net import QNet
from models.sac.policy_net import PolicyNet
from training.replay_buffer import ReplayBuffer


class SoftActorCritic:
    def __init__(self, args):
        learning_rate = args["learning_rate"]
        q_net_update_rate = args["q_net_update_rate"]
        batch_size = args["batch_size"]
        self.discount_factor = args["discount_factor"]

        # initialize two Q-networks and their respective target networks
        self.q_net_1 = QNet(learning_rate, q_net_update_rate)
        self.q_net_2 = QNet(learning_rate, q_net_update_rate)

        self.q_net_1_target = copy.deepcopy(self.q_net_1)
        self.q_net_2_target = copy.deepcopy(self.q_net_2)

        # get a replay buffer and a policy network
        self.memory = ReplayBuffer(
            buffer_limit=args["buffer_limit"], batch_size=batch_size)

        self.policy = PolicyNet(learning_rate=learning_rate,
                                entropy_alpha=args["entropy_alpha"])

    def train_step(self) -> Dict[str, float]:
        mini_batch = self.memory.sample()
        q_targets = self.calculate_q_targets(mini_batch)

        # update both q networks
        q_net_1_loss = self.q_net_1.train_step(q_targets, mini_batch)
        q_net_2_loss = self.q_net_2.train_step(q_targets, mini_batch)

        # update the policy
        policy_metrics = self.policy.train_step(
            self.q_net_1, self.q_net_2, mini_batch)

        # polyak updates for the target q networks
        self.q_net_1.polyak_update(self.q_net_1_target)
        self.q_net_2.polyak_update(self.q_net_2_target)

        return {"q_net_1_loss": q_net_1_loss,
                "q_net_2_loss": q_net_2_loss,
                **policy_metrics}

    def calculate_q_targets(self, mini_batch: tuple) -> torch.Tensor:
        _, _, rewards, next_states, dones = mini_batch

        with torch.no_grad():
            next_action, log_probabilities = self.policy(next_states)
            entropy = -self.policy.entropy_alpha * log_probabilities
            q1_val = self.q_net_1_target(next_states, next_action)
            q2_val = self.q_net_2_target(next_states, next_action)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = rewards + (1 - dones) * \
                self.discount_factor * (min_q + entropy)
        return target
