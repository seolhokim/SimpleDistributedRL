import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, config):
        self.state_shape = config.environment.state_shape
        self.action_dim = config.environment.action_dim
        self.unroll_length = config.training.unroll_length

        self.capacity = config.training.buffer_capacity
        self.device = config.training.device
        self.buffer_index = 0
        self.full = False
        self.buffer_initialize()

    def buffer_initialize(self):
        """Buffer initialization"""
        self.state = np.empty((self.capacity, self.unroll_length, *self.state_shape), dtype=np.float32)
        self.next_state = np.empty((self.capacity, self.unroll_length, *self.state_shape), dtype=np.float32)
        self.action = np.empty((self.capacity, self.unroll_length, self.action_dim), dtype=np.int64)
        self.action_prob = np.empty((self.capacity, self.unroll_length, 1), dtype=np.float32)
        self.reward = np.empty((self.capacity, self.unroll_length, 1), dtype=np.float32)
        self.terminated = np.empty((self.capacity, self.unroll_length, 1), dtype=bool)
        self.truncated = np.empty((self.capacity, self.unroll_length, 1), dtype=bool)

    def __len__(self):
        """Buffer length"""
        return self.capacity if self.full else self.buffer_index

    def add(self, data):
        """Add data into Buffer"""
        idx = self.buffer_index % self.capacity
        self.state[idx], self.action[idx], self.action_prob[idx], self.reward[idx], self.next_state[idx], self.terminated[idx], self.truncated[idx] = \
            data['state'], data['action'], data['action_prob'], data['reward'], data['next_state'], data['terminated'], data['truncated']

        self.buffer_index += 1
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size):
        """Sample data from Buffer"""
        indices = np.random.randint(0, len(self), size=batch_size) % self.capacity
        batch = {
            "states": torch.tensor(self.state[indices], device=self.device),
            "actions": torch.tensor(self.action[indices], dtype=torch.long, device=self.device),
            "action_probs": torch.tensor(self.action_prob[indices], device=self.device).squeeze(-1),
            "rewards": torch.tensor(self.reward[indices], device=self.device).squeeze(-1),
            "next_states": torch.tensor(self.next_state[indices], device=self.device),
            "terminated": torch.tensor(self.terminated[indices], dtype=torch.float, device=self.device).squeeze(-1),
            "truncated": torch.tensor(self.truncated[indices], device=self.device).squeeze(-1),
        }
        return batch


def make_actor_buffer(config):
    """
    Create a temporary buffer for actor data collection
    """
    buffer_dtypes = {
        "state": np.float32,
        "action": np.int32,
        "action_prob": np.float32,
        "reward": np.float32,
        "next_state": np.float32,
        "terminated": bool,
        "truncated": bool,
    }
    buffer_shapes = {
        "state": config.environment.state_shape,
        "action": (config.environment.action_dim,),
        "action_prob": (config.environment.action_dim,),
        "reward": (1,),
        "next_state": config.environment.state_shape,
        "terminated": (1,),
        "truncated": (1,),
    }

    actor_buffer = {
        key: np.empty(
            (config.training.unroll_length,) + shape, dtype=buffer_dtypes[key]
        )
        for key, shape in buffer_shapes.items()
    }
    return actor_buffer
