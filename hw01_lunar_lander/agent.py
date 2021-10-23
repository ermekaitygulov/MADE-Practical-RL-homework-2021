import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl").cpu()
        
    def act(self, state):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        state = torch.Tensor(state[None])
        with torch.no_grad():
            action = self.model(state).max(1)[1].numpy()
            action = action[0]
        return action

