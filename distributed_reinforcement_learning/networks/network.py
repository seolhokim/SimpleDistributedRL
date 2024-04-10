import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self, config):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(
                config.environment.state_shape[0], config.actor_network.hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(config.actor_network.hidden_dim, config.environment.n_action),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.network(state)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        dists = torch.distributions.Categorical(probs=probs)
        action = dists.sample()
        action_prob = probs.gather(1, action.unsqueeze(-1)).item()
        return action.item(), action_prob


class CriticNetwork(nn.Module):
    def __init__(self, config):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(
                config.environment.state_shape[0], config.critic_network.hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(config.critic_network.hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state)
