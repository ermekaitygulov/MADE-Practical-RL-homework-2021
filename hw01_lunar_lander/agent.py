import numpy as np
import torch
from network import NN_CATALOG


class Agent:
    def __init__(self):
        self.model = NN_CATALOG['DuelingDQN'](8, 4)
        model_weight = torch.load(__file__[:-8] + "/agent_weight.pkl")
        self.model.load_state_dict(model_weight)
        
    def act(self, state):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        state = torch.Tensor(state[None])
        with torch.no_grad():
            action = self.model(state).max(1)[1].numpy()
            action = action[0]
        return action

