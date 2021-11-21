import random
import numpy as np
import os
import torch
try:
    from train import Actor
except ModuleNotFoundError:
    from .train import Actor


class Agent:
    def __init__(self):
        self.model = Actor(28, 8)
        model_weight = torch.load(__file__[:-8] + "/actor_3254.15.pkl", map_location='cpu')
        self.model.load_state_dict(model_weight)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            action = self.model(state)
        return action.cpu().numpy()

    def reset(self):
        pass

