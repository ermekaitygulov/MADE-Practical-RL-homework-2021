import random
import numpy as np
import os
import torch
try:
    from train import Actor
except ModuleNotFoundError:
    from .train import Actor
from torch.nn import functional as F



class Agent:
    def __init__(self):
        self.model = Actor(22, 6)
        model_weight = torch.load(__file__[:-8] + "/agent_768.19.pkl", map_location='cpu')
        self.model.load_state_dict(model_weight)
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            _, _, distr = self.model.act(state)
            action = F.tanh(distr.mean)
            return action.cpu().numpy()[0]

    def reset(self):
        pass

