import time
import socket
import pickle
import threading
import torch.multiprocessing as mp

class LearnerManager:
    def __init__(self, ports, data_queue, weight_queue, data_queue_lock, weight_lock, replay_buffer, config, init_weight):
        self.data_queue = data_queue
        self.weight_queue = weight_queue
        self.data_queue_lock = data_queue_lock
        self.weight_lock = weight_lock
        self.replay_buffer = replay_buffer
        self.config = config
        self.weight = init_weight

        self.replay_buffer_lock = mp.Lock()
        self.ports = ports
        self.init_time = time.time()

    def start_servers(self):
        """Start servers on the specified ports to listen for incoming connections"""
        for port in self.ports:
            threading.Thread(target=self.start_server, args=(port,)).start()

    def start_server(self, port):
        """Start a server on the specified port to handle incoming connections"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(('', port))
            server_socket.listen()
            print(f"Listening for actors on port {port}...")
            client_socket, _ = server_socket.accept()
            self.handle_connection_with_actor(client_socket)

    def handle_connection_with_actor(self, client_socket):
        """Handle the connection from actor"""
        try:
            while True:
                socket_data = client_socket.recv(self.config.training.bufsize)
                if not socket_data:
                    break
                self.process_socket_data(client_socket, socket_data)
        finally:
            client_socket.close()

    def process_socket_data(self, client_socket, socket_data):
        """Process the received socket data"""
        message_type, message_data = pickle.loads(socket_data)
        if message_type == "data":
            self.data_to_replay_buffer(message_data)
        elif message_type == "get_weight":
            self.send_weight(client_socket)

    def data_to_replay_buffer(self, message_data):
        """Adding data to the replay buffer"""
        with self.replay_buffer_lock:
            self.replay_buffer.add(message_data)
            if len(self.replay_buffer) >= self.config.training.batch_size:
                buffer_fps = self.replay_buffer.buffer_index / (time.time() - self.init_time)
                batch_start_time = time.time()
                data = self.replay_buffer.sample(self.config.training.batch_size)
                self.try_put_data_queue(data, batch_start_time, buffer_fps * self.config.training.unroll_length)

    def send_weight(self, client_socket):
        """Sending the latest weights"""
        with self.weight_lock:
            if not self.weight_queue.empty():
                self.weight = self.weight_queue.get()
            client_socket.sendall(pickle.dumps(self.weight))

    def try_put_data_queue(self, data, batch_start_time, buffer_fps):
        """Try putting sampled data into the data queue"""
        with self.data_queue_lock:
            if self.data_queue.qsize() < self.config.training.max_queue_length:
                self.data_queue.put((data, time.time()-batch_start_time, buffer_fps))

    def start(self):
        """Entry point"""
        self.start_servers()
