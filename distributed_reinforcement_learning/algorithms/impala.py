import time
import torch
import torch.nn.functional as F

def train(
    num_train,
    data,
    metrics,
    actor,
    actor_optimizer,
    actor_lr_scheduler,
    critic,
    critic_optimizer,
    critic_lr_scheduler,
    clip_rho_threshold,
    clip_c_threshold,
    gamma,
    actor_network_max_norm,
    critic_network_max_norm,
    entropy_weight,
    epsilon,
):
    """impala algorithm to train ActorNetwork and CriticNetwork"""
    start_onestep_training_time = time.time()
    states, actions, behavior_action_probs, rewards, next_states, terminateds, truncateds = data["states"], data["actions"], data["action_probs"], data["rewards"], data["next_states"], data["terminated"], data["truncated"]
    target_action_probs = actor(states).gather(-1, actions).squeeze(-1).detach()
    
    importance_sampling_ratios = torch.exp(target_action_probs - behavior_action_probs)
    
    metrics['min_importance_sampling_ratio'] = importance_sampling_ratios.min().item()
    metrics['max_importance_sampling_ratio'] = importance_sampling_ratios.max().item()
    metrics['avg_importance_sampling_ratio'] = importance_sampling_ratios.mean().item()

    clipped_rho = torch.clamp(importance_sampling_ratios, min=clip_rho_threshold)
    clipped_c = torch.clamp(importance_sampling_ratios, min=clip_c_threshold)

    for i in range(num_train):
        start_forward_time = time.time()
        
        # Critic and actor values
        values = critic(states).squeeze()
        next_values = critic(next_states).squeeze()
        dists = torch.distributions.Categorical(probs=actor(states))
        log_probs = dists.log_prob(actions.squeeze())
        entropy = dists.entropy().mean()

        metrics['forward_time'] += time.time() - start_forward_time

        # vtrace corrections and critic update
        returns, advantages = get_vtrace_and_advantages(
            rewards,
            terminateds,
            truncateds,
            values,
            next_values,
            actions,
            clipped_rho,
            clipped_c,
            gamma,
            epsilon,
        )

        start_backward_time = time.time()
        
        critic_loss = 0.25 * F.mse_loss(values, returns.detach())
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=critic_network_max_norm)
        critic_optimizer.step()

        # Actor update
        actor_loss = -(clipped_rho.detach() * log_probs * advantages.detach()).mean() - entropy_weight * entropy
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=actor_network_max_norm)
        actor_optimizer.step()

        actor_lr_scheduler.step()
        critic_lr_scheduler.step()

        metrics['backward_time'] += time.time() - start_backward_time
        metrics['critic_loss'] += critic_loss.item()
        metrics['actor_loss'] += actor_loss.item()
        metrics['entropy'] += entropy.item()

    # Update metrics
    for key in ['critic_loss', 'actor_loss', 'entropy', 'forward_time', 'backward_time']:
        metrics[key] /= num_train
    

    metrics['onestep_training_time'] = time.time() - start_onestep_training_time

    return metrics


def get_vtrace_and_advantages(
    rewards,
    terminateds,
    truncateds,
    values,
    next_values,
    actions,
    clipped_rho,
    clipped_c,
    gamma,
    epsilon,
):
    """Computes v-trace targets and advantages"""
    batch_size, T = rewards.shape
    vtraces = torch.zeros_like(rewards, device=rewards.device)
    advantages = torch.zeros_like(rewards, device=rewards.device)

    next_vtrace = next_values[:, -1]


    for t in reversed(range(T)):
        next_vtrace = torch.where(truncateds[:, t], next_values[:, t], next_vtrace)
        delta = clipped_rho[:,t] * (rewards[:, t] + gamma * next_values[:, t] * (1 - terminateds[:, t]) - values[:, t])
        vtraces[:, t] = values[:, t] + delta + gamma * clipped_c[:,t] * (next_vtrace - next_values[:, t]) * (1 - terminateds[:, t])
        
        advantages[:, t] = rewards[:, t] + gamma * next_vtrace * (1 - terminateds[:, t]) - values[:, t]

        next_vtrace = vtraces[:, t]

    return vtraces, advantages
