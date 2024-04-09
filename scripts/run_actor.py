import sys
from pathlib import Path                            ##this is for one machine
impala_dir = Path(__file__).resolve().parent.parent ##this is for one machine
sys.path.append(str(impala_dir))                    ##this is for one machine

import socket
from distributed_reinforcement_learning.configs.base_config import Config as config
from distributed_reinforcement_learning.networks.network import ActorNetwork, CriticNetwork
from distributed_reinforcement_learning.agents.actors.actor import Actor
from distributed_reinforcement_learning.utils.replay_buffer import make_actor_buffer


def connect_to_learner_manager(port, actor_log_path):
    """Initializes actor and connect to the LearnerManger"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', port))
        print(f"Connected to port {port}")

        actor_network = ActorNetwork(config)
        actor_buffer = make_actor_buffer(config)

        actor = Actor(s, actor_network, actor_buffer, config, actor_log_path+f"/{str(port)}")
        actor.start()

if __name__ == "__main__":
    if len(sys.argv) < 2: # need to run it with port
        sys.exit(1)
    
    port = int(sys.argv[1])
    actor_log_path = str(sys.argv[2])
    connect_to_learner_manager(port, actor_log_path)
