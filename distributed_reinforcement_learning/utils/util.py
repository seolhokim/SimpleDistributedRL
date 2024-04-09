import torch.multiprocessing as mp
from datetime import datetime
import subprocess

def generate_experiment_name(module):
    """Generates a unique name based on module"""
    module_name = module.__module__.split('.')[-1]
    datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result = f"runs/{module_name}_{datetime_str}"
    return result

def run_actor_scripts(ports, experiment_name):
    """Launches run_actor.py scripts on different port"""
    for port in ports:
        subprocess.Popen(["python", "scripts/run_actor.py", str(port), experiment_name])

def get_queue_and_lock():
    """Creates Queue and a Lock"""
    return mp.Queue(), mp.Lock()