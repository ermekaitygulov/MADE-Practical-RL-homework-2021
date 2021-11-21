from gym import make

from agent import Agent
from train import ENV_NAME
import numpy as np


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


if __name__ == '__main__':
    env = make(ENV_NAME)
    agent = Agent()
    rewards = evaluate_policy(env, agent, 50)
    print(f"Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")