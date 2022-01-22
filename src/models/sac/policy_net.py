import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Tuple, Dict, List
from torch.distributions import Normal


class PolicyNet(nn.Module):
    def __init__(self, learning_rate: float, entropy_alpha: float):
        # make sure that the policy network is registered as a pytorch module
        super(PolicyNet, self).__init__()

        # specify neurons per layer. "fc" is short for "fully_connected layer".
        self.common_mlp = nn.Linear(3, 128)
        self.mean_mlp = nn.Linear(128, 1)
        self.std_mlp = nn.Linear(128, 1)

        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate)  # use adam optimizer

        self.entropy_alpha = entropy_alpha  # weight of the entropy term

    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Choose an action and calcualte its probability for the current state.
        :param state: The state to choose the action for
        :param deterministic: Whether to draw an action from
        :return:
        """
        state = F.relu(self.common_mlp(state))
        mean = self.mean_mlp(state)
        # we need the standard deviation to be >0
        std = F.softplus(self.std_mlp(state))
        normal_distribution = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = normal_distribution.rsample()
        log_probabilities = normal_distribution.log_prob(action)

        # the original SAC implementation also squishes the action into [-1, 1] using a tanh activation.
        # to keep the probabilities correct, they account for this using the update below.

        real_action = torch.tanh(action)
        real_log_probabilities = log_probabilities - \
            torch.log(1 - torch.tanh(action).pow(2) + 1e-7)

        return real_action, real_log_probabilities

    def train_step(self, q_net_1, q_net_2, mini_batch: tuple) -> Dict[str, float]:
        states, _, _, _, _ = mini_batch
        actions, log_probabilities = self.forward(states)
        entropy = -self.entropy_alpha * log_probabilities

        # evaluate both q-networks for the current state-action pair
        # and use their minimum (see Twin-Delayed Q functions)

        q1_value = q_net_1(states, actions)
        q2_value = q_net_2(states, actions)
        q1_q2 = torch.cat([q1_value, q2_value], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -(min_q + entropy).mean()  # "-" for gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"entropy": entropy.mean().item(),
                "policy_loss": loss.item()}
