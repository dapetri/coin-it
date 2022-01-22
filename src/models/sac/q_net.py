import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, learning_rate: float, q_net_update_rate: float):
        # make sure that the Q Network is registered as a pytorch module
        super(QNet, self).__init__()
        # specify network parameters. "fc" is short for "fully_connected layer".
        self.state_layer = nn.Linear(3, 64)
        self.action_layer = nn.Linear(1, 64)
        self.common_layer = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate)  # use adam optimizer
        self.q_net_update_rate = q_net_update_rate

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.state_layer(state))
        h2 = F.relu(self.action_layer(action))
        cat = torch.cat([h1, h2], dim=1)
        q_evaluation = F.relu(self.common_layer(cat))
        q_evaluation = self.fc_out(q_evaluation)
        return q_evaluation

    def train_step(self, target_values: torch.Tensor, mini_batch: tuple) -> float:
        """
        Train the network for a single mini-batch update
        :param target_values: The target values to regress to
        :param mini_batch: A tuple (state, action, reward, next_state, done). For this update,
          only the action and state are needed
        :return: The mean loss for this update step
        """
        states, actions, _, _, _ = mini_batch  # get action and state from current mini_batch
        evaluation = self.forward(states, actions)

        # calculate the loss and its gradients; update the network based on them
        loss = F.smooth_l1_loss(evaluation, target_values).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def polyak_update(self, target_network):
        """
        Soft update the target network with the parameters of this network
        :param target_network:
        :return:
        """
        for param_target, param in zip(target_network.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.q_net_update_rate)
                                    + param.data * self.q_net_update_rate)
