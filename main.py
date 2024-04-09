import torch.multiprocessing as mp
from distributed_reinforcement_learning.configs.base_config import Config as config
from distributed_reinforcement_learning.networks.network import ActorNetwork, CriticNetwork
from distributed_reinforcement_learning.utils.replay_buffer import ReplayBuffer
from distributed_reinforcement_learning.agents.learners.learner import Learner
from distributed_reinforcement_learning.agents.learners.learner_manager import LearnerManager
from distributed_reinforcement_learning.utils.util import (
    generate_experiment_name, run_actor_scripts, get_queue_and_lock
)

def setup_config(config):
    """Set up configuration"""
    return generate_experiment_name(config)

def setup_networks(config):
    """Set up actor and critic networks"""
    actor_network = ActorNetwork(config)
    critic_network = CriticNetwork(config)
    return actor_network, critic_network

def setup_multiprocessing():
    """Set up multiprocessing configuration"""
    data_queue, data_queue_lock = get_queue_and_lock()
    weight_queue, weight_lock = get_queue_and_lock()
    return data_queue, data_queue_lock, weight_queue, weight_lock

def setup_learner_manager(config, experiment_name, actor_network, critic_network, data_queue, weight_queue, data_queue_lock, weight_lock):
    """Set up the Learner and Learner Manager"""
    init_weight = actor_network.state_dict()
    replay_buffer = ReplayBuffer(config)
    learner_log_path = f"{experiment_name}_Learner"
    learner = Learner(
        learner_log_path, data_queue, weight_queue, data_queue_lock, weight_lock, actor_network, critic_network, config
    )
    learner_manager = LearnerManager(
        config.training.ports, data_queue, weight_queue, data_queue_lock, weight_lock, replay_buffer, config, init_weight
    )
    return learner, learner_manager

def main():
    mp.set_start_method('spawn', force=True)
    
    experiment_name = setup_config(config)
    actor_network, critic_network = setup_networks(config)
    data_queue, data_queue_lock, weight_queue, weight_lock = setup_multiprocessing()
    learner, learner_manager = setup_learner_manager(
        config, experiment_name, actor_network, critic_network, data_queue, weight_queue, data_queue_lock, weight_lock
    )

    learner_manager.start()
    learner.start()

    actor_log_path = f"{experiment_name}_Actors"
    run_actor_scripts(config.training.ports, actor_log_path)

if __name__ == '__main__':
    main()
