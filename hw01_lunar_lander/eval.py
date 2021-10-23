from agent import Agent
from train import evaluate_policy
import numpy as np


if __name__ == '__main__':
    agent = Agent()
    rewards = evaluate_policy(agent, 50)
    print(f"Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")