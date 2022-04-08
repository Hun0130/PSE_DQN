import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

# ========================= Hyper Parameter ========================
learning_rate = 0.005
gamma         = 0.98
buffer_limit  = 5000
batch_size    = 32
epoch = 10000
train_interval = 'episode original'
update_interval = 20
# ========================= Hyper Parameter ========================

# ReplayBuffer
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen = buffer_limit)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst = []
        a_lst = []
        r_lst = [] 
        s_prime_lst = []
        done_mask_lst = []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)         
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(np.array(s_lst), dtype = torch.float, device = self.device), torch.tensor(np.array(a_lst), 
                device = self.device), torch.tensor(np.array(r_lst), dtype = torch.float, device = self.device), \
                torch.tensor(np.array(s_prime_lst), dtype=torch.float, device = self.device),\
                torch.tensor(np.array(done_mask_lst), dtype = torch.float, device = self.device)
    
    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

# Neural Network Model Name: Qnet
class Qnet(nn.Module):
    def __init__(self, input, output):
        super(Qnet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dnn = nn.Sequential(
        nn.Linear(input, 1240),
        nn.ReLU(),
        nn.Linear(1240, 930),
        nn.ReLU(),
        nn.Linear(930, 610),
        nn.ReLU(),
        nn.Linear(610, 320),
        nn.ReLU(),
        nn.Linear(320, 320)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, output)
        )

    def forward(self, x):
        x = torch.Tensor.clone(x).detach().to('cuda')
        features = self.dnn(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean())
        return q_vals
    
    def dist(self, x, y):
        result = []
        for i in range(len(x)):
            result.append((float(x[i])-float(y[i]))**2)
        return sum(result)
    
    def sample_action(self, obs, epsilon, choice, stock):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon :
            return random.randint(0, len(choice) - 1)
        else : 
            for i in range(len(choice)):
                model = choice[i][0][0]
                for j in stock.items():
                    if j[0] == model:
                        if j[1] <= 0:
                            out[i] = -99999
            return out.argmax().item()
        
    def save(self, file_name = 'model.pth'):
        model_folder_path = 'Dueling_DQN_model/'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

def train(q, q_target, memory, optimizer):
    result = 0
    # for i in range(10):
    s,a,r,s_prime,done_mask = memory.sample(batch_size)
    # current Q values
    curr_Q = q(s).gather(1, a)
    # curr_Q = curr_Q.squeeze(1)
    
    # Q prime values
    max_next_Q = q_target(s_prime).max(1)[0].unsqueeze(1)
    expected_Q = r + gamma * max_next_Q  * done_mask
    

    loss = F.smooth_l1_loss(curr_Q, expected_Q)
    result += loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return result

def train_long(q, q_target, memory, optimizer):
    result = 0
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        # current Q values
        curr_Q = q(s).gather(1, a)
        # curr_Q = curr_Q.squeeze(1)
        
        # Q prime values
        max_next_Q = q_target(s_prime).max(1)[0].unsqueeze(1)
        expected_Q = r + gamma * max_next_Q  * done_mask
        

        loss = F.smooth_l1_loss(curr_Q, expected_Q)
        result += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return result