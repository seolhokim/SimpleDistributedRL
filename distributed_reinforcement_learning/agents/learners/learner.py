from distributed_reinforcement_learning.networks.network import ActorNetwork, CriticNetwork
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import torch.multiprocessing as mp
import torch.optim as optim
import torch
import time


class Learner(mp.Process):
    def __init__(
        self,
        writer_path,
        data_queue,
        weight_queue,
        data_queue_lock,
        weight_lock,
        actor_network,
        critic_network,
        config,
    ):
        super().__init__()
        self.writer_path = writer_path
        self.config = config
        self.device = torch.device(config.training.device)
        self.setup_networks(actor_network, critic_network)
        self.data_queue = data_queue
        self.weight_queue = weight_queue
        self.data_queue_lock = data_queue_lock
        self.weight_lock = weight_lock
        self.train_count = 0

    def setup_networks(self, actor_network, critic_network):
        """Initialize and configure networks and optimizers"""
        self.actor_network = actor_network.to(self.device)
        self.critic_network = critic_network.to(self.device)
        self.actor_network_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.config.actor_network.learning_rate)
        self.critic_network_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.config.critic_network.learning_rate)
        #self.actor_network_optimizer = optim.RMSprop(self.actor_network.parameters(), lr=1e-3, momentum=0.0, eps=0.01)
        #self.critic_network_optimizer = optim.RMSprop(self.actor_network.parameters(), lr=1e-3, momentum=0.0, eps=0.01)
        self.setup_schedulers()

    def setup_schedulers(self):
        """Set up learning rate schedulers"""
        self.actor_network_lr_scheduler = optim.lr_scheduler.StepLR(self.actor_network_optimizer, step_size=self.config.actor_network.step_size, gamma=self.config.actor_network.gamma)
        self.critic_network_lr_scheduler = optim.lr_scheduler.StepLR(self.critic_network_optimizer, step_size=self.config.critic_network.step_size, gamma=self.config.critic_network.gamma)

    def training(self):
        """The main training loop"""
        writer = SummaryWriter(self.writer_path)
        while True:
            metrics = defaultdict(int)
            data, metrics = self.fetch_data(metrics)
            if data:
                self.perform_training_iteration(writer, data, metrics)

    def perform_training_iteration(self, writer, data, metrics):
        """Main training loop"""
        metrics = self.config.training.algorithm(
            self.config.training.num_train,
            data,
            metrics,
            self.actor_network,
            self.actor_network_optimizer,
            self.actor_network_lr_scheduler,
            self.critic_network,
            self.critic_network_optimizer,
            self.critic_network_lr_scheduler,
            clip_rho_threshold=self.config.impala.clip_rho_threshold,
            clip_c_threshold=self.config.impala.clip_c_threshold,
            gamma=self.config.impala.gamma,
            actor_network_max_norm=self.config.actor_network.max_norm,
            critic_network_max_norm=self.config.critic_network.max_norm,
            entropy_weight=self.config.impala.entropy_weight,
            baseline_loss_scaling = self.config.training.baseline_loss_scaling,
            epsilon=self.config.impala.epsilon,
        )

        self.log_metrics(writer, metrics)
        self.send_new_weight()

    def fetch_data(self, metrics):
        """fetch data from the data queue"""
        with self.data_queue_lock:
            if not self.data_queue.empty():
                data_fetch_start_time = time.time()
                data = self.data_queue.get()
                data, data_batch_time, buffer_fps = data
                metrics['data_fetch_time'] = time.time() - data_fetch_start_time
                metrics['data_batch_time'] = data_batch_time
                metrics['buffer_fps'] = buffer_fps
                return data, metrics
        return None, None #data, metrics

    def log_metrics(self, writer, metrics):
        """Log metrics"""
        for key, value in metrics.items():
            writer.add_scalar(f'Learner/{key}', value, self.train_count)
        self.train_count += 1

    def send_new_weight(self):
        """Send new weights to the weight queue"""
        state_dict = {k: v.cpu().detach() for k, v in self.actor_network.state_dict().items()}
        with self.weight_lock:
            self.clear_queue(self.weight_queue) 
            self.weight_queue.put(state_dict)

    def clear_queue(self, queue):
        """Clear all items"""
        while not queue.empty():
            queue.get()

    def run(self):
        """Entry point for the multiprocessing"""
        self.training()
