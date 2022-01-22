import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, buffer_limit: int, batch_size: int):
        self.batch_size = batch_size
        self.buffer = collections.deque(
            maxlen=buffer_limit)  # use a dequeue as a buffer

    def put(self, transition: tuple) -> None:
        """
        Adds a transition to the buffer.
        :param transition: (s, a, r, s', done) pair sampled by having the policy act on the environment
        :return: None
        """
        self.buffer.append(transition)

    def sample(self) -> tuple:
        # get self.batch_size random samples from the buffer
        mini_batch = random.sample(self.buffer, self.batch_size)
        # initialize list of (s, a, r, s', done) tuples
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transition in mini_batch:  # parse all transitions into their lists.
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            dones.append([float(done)])

        return torch.tensor(states, dtype=torch.float), \
            torch.tensor(actions, dtype=torch.float), \
            torch.tensor(rewards, dtype=torch.float), \
            torch.tensor(next_states, dtype=torch.float), \
            torch.tensor(dones, dtype=torch.float)

    def size(self) -> int:
        return len(self.buffer)
