# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:21:46 2020

@author: KJP
"""
import gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([2,15,5,18]).cuda()
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).cuda()
        return x
      
    def sample_action(self, obs, epsilon):
        print("obs: ",obs)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            print(out)
            return out.argmax().item()
        

q = Qnet()
q_target = Qnet()
q_target.load_state_dict(q.state_dict())

a = q.sample_action(torch.from_numpy(x).float(), 0)
print(random.randint(0,2))