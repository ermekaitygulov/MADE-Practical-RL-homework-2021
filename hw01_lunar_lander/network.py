import random

import torch
import torch.nn as nn
import torch.nn.functional as F

NN_CATALOG = {}


def add_to_catalog(name, catalog):
    def add_wrapper(class_to_add):
        catalog[name] = class_to_add
        return class_to_add
    return add_wrapper


@add_to_catalog('DQN', NN_CATALOG)
class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim, n_layers=3):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_layers = n_layers

        self.nn = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        for i in range(min(n_layers - 2, 0)):
            self.nn.add_module(f'{i}_hid', nn.Linear(128, 128))
            self.nn.add_module(f'{i}_relu', nn.ReLU())
        self.nn.add_module('out', nn.Linear(128, act_dim))

    def forward(self, state):
        # src = [src sent len, batch size]
        output = self.nn(state)
        return output


@add_to_catalog('DuelingDQN', NN_CATALOG)
class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.nn_in = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.a_hid = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        self.v_hid = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        # src = [src sent len, batch size]
        hid = self.nn_in(state)
        a_out = self.a_hid(hid)
        v_out = self.v_hid(hid)
        a_out = a_out - torch.mean(a_out, 1, keepdim=True)
        q_out = v_out + a_out
        return q_out
