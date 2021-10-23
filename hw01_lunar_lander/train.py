from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

from network import NN_CATALOG

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NETWORK_NAME = 'DuelingDQN'
np.random.seed(42)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0 # Do not change
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NN_CATALOG[NETWORK_NAME](state_dim, action_dim)
        self.model.to(self.device)
        self.target_net = NN_CATALOG[NETWORK_NAME](state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()
        self.optim = Adam(self.model.parameters())
        self.buffer = deque(maxlen=int(1e6))
        self.criterion = nn.MSELoss()

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch_idx = [random.randint(0, len(self.buffer) - 1) for _ in range(BATCH_SIZE)]
        batch_keys = ['state', 'action', 'next_state', 'reward', 'done']
        batch = []
        for key_i, key in enumerate(batch_keys):
            batch.append(np.array([self.buffer[idx][key_i] for idx in batch_idx]))
        return batch
        
    def train_step(self, batch):
        # Use batch to update DQN's network.
        tensor_batch = self.prepare_batch(batch)
        s_batch, a_batch, ns_batch, r_batch, d_batch = tensor_batch
        q_batch = self.model(s_batch).gather(-1, a_batch[:, None].long()).squeeze()
        na_batch = self.model(ns_batch).max(1)[1].detach()
        target = self.target_net(ns_batch).gather(-1, na_batch[:, None].long()).squeeze()
        # target = self.target_net(ns_batch).max(1)[0].detach()
        target *= (1 - d_batch.float()) * GAMMA
        target += r_batch
        target = target.detach()
        loss = self.criterion(q_batch, target)

        self.optim.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target_net.load_state_dict(self.model.state_dict())

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        state = torch.Tensor(state[None]).to(self.device)
        with torch.no_grad():
            action = self.model(state).max(1)[1].cpu().numpy()
            action = action[0]
        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")

    def prepare_batch(self, batch):
        tensor_batch = []
        for b in batch:
            tensor_batch.append(torch.Tensor(b).to(self.device))
        return tensor_batch


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
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
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 10)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
        if (i + 1) % 10000 == 0:
            eps = max(0.01, eps * 0.9)
