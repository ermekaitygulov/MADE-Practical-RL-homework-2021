from agent import Agent
from train import evaluate_policy
import numpy as np


if __name__ == '__main__':
    rewards = evaluate_policy(Agent(), 50)
    print(f"Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
