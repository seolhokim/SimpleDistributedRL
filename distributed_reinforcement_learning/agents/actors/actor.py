from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import time
import gym
import pickle

class Actor(mp.Process):
    def __init__(self, conn, actor_network, actor_buffer, config, actor_log_path = "./runs/Actor"):
        super().__init__()
        self.conn = conn
        self.conn.setblocking(0) ## check result

        self.config = config
        self.env = gym.make(self.config.environment.env_name)
        self.actor_network = actor_network 
        self.actor_buffer = actor_buffer

        self.last_weight_request_time = time.time()
        self.weight_update_interval = 0
        self.got_weight = True
        self.logging_count = 0 
        self.interaction_count = 0
        self.buffer_size = self.config.training.unroll_length
        self.writer_path = actor_log_path
        self.cur_score = 0

    def run(self):
        """Entry point"""
        writer = SummaryWriter(self.writer_path)
        self.run_actor(writer)

    def run_actor(self, writer):
        """Main loop for the actor to interact with the environment"""
        data_sending_time = 0
        start_time = time.time()
        for episode in range(self.config.training.num_interaction_episodes):

            state, _ = self.env.reset()
            terminated, truncated = False, False
            cur_score = 0

            while not (terminated or truncated):
                action, action_prob = self.actor_network.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                self.update_actor_buffer(state, action, action_prob, reward, next_state, terminated, truncated)
                cur_score += reward
                state = next_state
                
                if self.interaction_count % self.buffer_size == 0 and self.interaction_count != 0:
                    data_sending_time = self.send_data()
                    if self.should_request_weight():
                        self.request_weight()
                    self.get_weight()
            self.cur_score = cur_score
            self.log_metrics(writer, start_time, data_sending_time)

            if episode % self.config.training.mean_interval == 0:
                print(f"actor Episode {episode} completed, cur_score : {self.cur_score} ")
        
    def update_actor_buffer(self, state, action, action_prob, reward, next_state, terminated, truncated):
        """Updates the experience buffer"""
        buffer_idx = self.interaction_count % self.buffer_size
        self.actor_buffer["state"][buffer_idx] = state
        self.actor_buffer["action"][buffer_idx] = action
        self.actor_buffer["action_prob"][buffer_idx] = action_prob
        self.actor_buffer["reward"][buffer_idx] = reward
        self.actor_buffer["next_state"][buffer_idx] = next_state
        self.actor_buffer["terminated"][buffer_idx] = terminated
        self.actor_buffer["truncated"][buffer_idx] = truncated
        self.interaction_count += 1

    def should_request_weight(self):
        """Check it's time to request new weights"""
        return self.got_weight and (time.time() - self.last_weight_request_time) >= self.config.training.update_interval

    def send_data(self):
        """Sends the collected data"""
        data_sending_start_time = time.time()
        data = pickle.dumps(("data", self.actor_buffer)) #non-blocking test
        self.conn.sendall(data)
        return time.time() - data_sending_start_time

    def request_weight(self):
        """Sends a request to the learner"""
        self.conn.sendall(pickle.dumps(("get_weight", None)))
        self.last_weight_request_time = time.time()
        self.got_weight = False

    def log_metrics(self, writer, start_time, data_sending_time):
        """Log metrics"""
        fps = self.interaction_count / (time.time() - start_time)
        writer.add_scalar('Actor/score', self.cur_score, self.logging_count)
        writer.add_scalar('Actor/fps', fps, self.logging_count)
        writer.add_scalar('Actor/data_sending_time', data_sending_time, self.logging_count)
        writer.add_scalar('Actor/weight_update_interval', self.weight_update_interval, self.logging_count)
        self.logging_count += 1

    def get_weight(self):
        """Attempts to receive new model weights"""
        try:
            new_weights = pickle.loads(self.conn.recv(self.config.training.bufsize))
            self.actor_network.load_state_dict(new_weights)
            self.got_weight = True
            self.weight_update_interval = time.time() - self.last_weight_request_time
        except BlockingIOError:
            #no data to receive
            pass