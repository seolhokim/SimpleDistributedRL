class ActorNetworkConfig:
    hidden_dim = 128
    learning_rate = 1e-3
    step_size = 50
    gamma = 0.9
    max_norm = 0.1


class CriticNetworkConfig:
    hidden_dim = 128
    learning_rate = 1e-3
    step_size = 50
    gamma = 0.9
    max_norm = 0.1


class EnvironmentConfig:
    env_name = "CartPole-v1"
    state_shape = (4,)
    action_dim = 1
    n_action = 2

from distributed_reinforcement_learning.algorithms.impala import train
class TrainingConfig:
    algorithm = train
    unroll_length = 100
    update_interval = 0.5
    bufsize = 8192
    num_interaction_episodes = 10000
    buffer_capacity = 1000
    batch_size = 16
    num_actors = 4
    mean_interval = 10
    device = "cuda" 
    ports = [5000,5001,5002,5003]
    num_train = 3
    max_queue_length = 20

class ImpalaConfig:
    clip_rho_threshold = 1.0
    clip_c_threshold = 1.0
    gamma = 0.99
    entropy_weight = 0.01
    epsilon = 1e-6


class Config:
    actor_network = ActorNetworkConfig
    critic_network = CriticNetworkConfig
    environment = EnvironmentConfig
    training = TrainingConfig
    impala = ImpalaConfig
