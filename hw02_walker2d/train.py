from collections import defaultdict

import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import wandb
import random

from util import WalkerRewardShape

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 2e-4
CRITIC_LR = 1e-3

CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 64
BATCH_SIZE = 64

MIN_TRANSITIONS_PER_UPDATE = 4096 #2048
MIN_EPISODES_PER_UPDATE = 10 #4
ITERATIONS = 1000

KL_MAX = 1
LOAD_FROM = {
    'actor_path': 'actor_660.55.pkl',
    'critic_path': 'critic_660.55.pkl',
}

CONFIG = dict(
    LAMBDA=LAMBDA,
    GAMMA=GAMMA,
    ACTOR_LR=ACTOR_LR,
    CRITIC_LR=CRITIC_LR,
    CLIP=CLIP,
    ENTROPY_COEF=ENTROPY_COEF,
    BATCHES_PER_UPDATE=BATCHES_PER_UPDATE,
    BATCH_SIZE=BATCH_SIZE,
    MIN_TRANSITIONS_PER_UPDATE=MIN_TRANSITIONS_PER_UPDATE,
    MIN_EPISODES_PER_UPDATE=MIN_EPISODES_PER_UPDATE,
    ITERATIONS=ITERATIONS,
    KL_MAX=KL_MAX
)

    
def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv, v_old) for (s, a, _, p, v_old), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        self.sigma = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        
    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        return None
        
    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        action_mean = self.model(state)
        action_distr = Normal(action_mean, torch.exp(self.sigma))
        raw_action = action_distr.sample()
        action = F.tanh(raw_action)
        return action, raw_action, action_distr
        
        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR, weight_decay=1e-5)

    def update(self, trajectories, steps_done):
        transitions = [t for traj in trajectories for t in traj] # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage, value_old = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        value_old = np.array(value_old)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        log_dict = defaultdict(list)
        
        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE) # Choose random batch
            s = torch.tensor(state[idx]).float().to(self.device)
            a = torch.tensor(action[idx]).float().to(self.device)
            op = torch.tensor(old_prob[idx]).float().to(self.device) # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float().to(self.device) # Estimated by lambda-returns
            v_old = torch.tensor(value_old[idx]).float().to(self.device)
            adv = torch.tensor(advantage[idx]).float().to(self.device) # Estimated by generalized advantage estimation
            _, _, policy = self.actor.act(s)
            value = self.critic.get_value(s).squeeze()
            # value = v_old + (value - v_old).clamp(-CLIP, CLIP)

            p = policy.log_prob(a).sum(-1)
            p_odds = torch.exp(p - op)
            g = torch.where(adv >= 0, 1 + CLIP, 1 - CLIP)
            policy_gain = torch.minimum(p_odds * adv, g * adv).mean()
            entropy = policy.entropy().mean()
            actor_loss = - policy_gain - entropy * ENTROPY_COEF

            val_loss = F.mse_loss(value.squeeze(), v)
            critic_loss = val_loss

            # TODO: Update actor here
            log_dict['kl_div'].append(F.kl_div(p, op, log_target=True).cpu().item())
            if log_dict['kl_div'][-1] > KL_MAX:
                break
            log_dict['actor_grad'].append(self.update_nn(self.actor_optim, actor_loss, self.actor))
            log_dict['critic_grad'].append(self.update_nn(self.critic_optim, critic_loss, self.critic))

            log_dict['policy_gain'].append(policy_gain.cpu().item())
            log_dict['entropy'].append(entropy.cpu().item())
            log_dict['val_loss'].append(val_loss.cpu().item())

            # TODO: Update critic here
        if wandb.run:
            wandb.log({'step': steps_done, **{key: np.mean(value) for key, value in log_dict.items()}})
        else:
            print(f'{key}: {np.mean(value):.3f}' for key, value in log_dict.items())

    @staticmethod
    def update_nn(optim, loss, model):
        optim.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)

        optim.step()

        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(self.device)
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(self.device)
            action, pure_action, distr = self.actor.act(state)
            prob = distr.log_prob(pure_action).sum(-1)
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self, reward):
        torch.save(self.actor.state_dict(), f"actor_{reward:.2f}.pkl")
        torch.save(self.critic.state_dict(), f"critic_{reward:.2f}.pkl")

    def load(self, actor_path, critic_path):
        if actor_path:
            actor_weight = torch.load(actor_path, map_location=self.device)
            self.actor.load_state_dict(actor_weight)

        if critic_path:
            critic_weight = torch.load(critic_path, map_location=self.device)
            self.critic.load_state_dict(critic_weight)


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)


if __name__ == "__main__":
    try:
        wandb.init(
            entity='ermekaitygulov',
            anonymous='allow',
            project='RL-HW2',
            force=False,
            config=CONFIG
        )
    except wandb.errors.error.UsageError:
        pass

    env = make(ENV_NAME)
    train_env = WalkerRewardShape(env)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    ppo.load(**LOAD_FROM)
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    
    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0
        
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(train_env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories, steps_sampled)
        
        if (i + 1) % (ITERATIONS//100) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            if np.mean(rewards) > 500:
                ppo.save(np.mean(rewards))
