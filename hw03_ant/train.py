import pybullet_envs
from gym import make
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy
import wandb

GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 1000000
DELAY = 4
POLICY_NOISE = 0.05
NOISE_CLIP = 0.2
EPSILON = 0.1
USE_HUBER = True

CONFIG = {
    key: value for key, value in locals().copy().items() if key.isupper() and not str(value).startswith("<module ")
}


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.model(state)
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=CRITIC_LR)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.total_it = 0
        self.replay_buffer = deque(maxlen=200000)

    def update(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            self.total_it += 1
            log_dict = dict()
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer)-1)] for _ in range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)
            
            # Update critic
            log_dict['critic'] = self.update_critic(state, action, reward, next_state, done)
            
            # Update actor
            if (self.total_it % DELAY) == 0:
                log_dict['actor'] = self.update_actor(state)

                soft_update(self.target_critic_1, self.critic_1)
                soft_update(self.target_critic_2, self.critic_2)
                soft_update(self.target_actor, self.actor)
            if wandb.run:
                wandb.log({'optim_step': self.total_it, **log_dict})

    def update_critic(self, state, action, reward, next_state, done):
        q_1 = self.critic_1(state, action)
        q_2 = self.critic_2(state, action)
        log_dict = dict()
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            noise = (
                    torch.randn_like(action) * POLICY_NOISE
            ).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (next_action + noise).clamp(-1, 1)

            next_q_1 = self.target_critic_1(next_state, next_action)
            next_q_2 = self.target_critic_2(next_state, next_action)

            target = torch.minimum(next_q_1, next_q_2) * GAMMA
            target = target * (1 - done)
            target += reward

        if USE_HUBER:
            critic_loss1 = F.smooth_l1_loss(q_1, target)
            critic_loss2 = F.smooth_l1_loss(q_2, target)
        else:
            critic_loss1 = F.mse_loss(q_1, target)
            critic_loss2 = F.mse_loss(q_2, target)
        log_dict['critic_loss1'] = critic_loss1.cpu().item()
        log_dict['critic_loss2'] = critic_loss2.cpu().item()

        log_dict['critic1_grad'] = self.update_nn(self.critic_1_optim, critic_loss1, self.critic_1)
        log_dict['critic2_grad'] = self.update_nn(self.critic_2_optim, critic_loss2, self.critic_2)
        return log_dict

    def update_actor(self, state):
        log_dict = dict()
        pred_action = self.actor(state)
        q_value = self.critic_1(state, pred_action)
        q_mean = -torch.mean(q_value)
        actor_loss = q_mean
        log_dict['actor_loss'] = actor_loss.cpu().item()

        log_dict['actor_grad'] = self.update_nn(self.actor_optim, actor_loss, self.actor)
        return log_dict

    @staticmethod
    def update_nn(optim, loss, model):
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()[0]

    def save(self, reward):
        torch.save(self.actor.state_dict(), f"actor_{reward:.2f}.pkl")
        torch.save(self.critic_1.state_dict(), f"critic_{reward:.2f}.pkl")

    def load(self, actor_path, critic_path):
        if actor_path:
            actor_weight = torch.load(actor_path, map_location=DEVICE)
            self.actor.load_state_dict(actor_weight)

        if critic_path:
            critic_weight = torch.load(critic_path, map_location=DEVICE)
            self.critic_1.load_state_dict(critic_weight)
            self.critic_2.load_state_dict(critic_weight)

            self.target_critic_1 = copy.deepcopy(self.critic_1)
            self.target_critic_2 = copy.deepcopy(self.critic_2)


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    try:
        wandb.init(
            entity='ermekaitygulov',
            anonymous='allow',
            project='RL-HW3',
            force=False,
            config=CONFIG
        )
    except wandb.errors.error.UsageError:
        pass
    env = make(ENV_NAME)
    test_env = make(ENV_NAME)
    td3 = TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()

    episodes_sampled = 0
    steps_sampled = 0

    max_reward = -np.inf
    train_reward = 0

    for i in range(TRANSITIONS):
        steps = 0
        
        #Epsilon-greedy policy
        action = td3.act(state)
        action = np.clip(action + EPSILON * np.random.randn(*action.shape), -1, +1)

        next_state, reward, done, _ = env.step(action)
        td3.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()
        train_reward += reward
        if done:
            wandb.log({'train_step': i + 1, 'train_reward': train_reward})
            train_reward = 0

        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(test_env, td3, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            wandb.log({'train_step': i + 1, 'val': np.mean(rewards)})
            if np.mean(rewards) > max_reward:
                td3.save(np.mean(rewards))

