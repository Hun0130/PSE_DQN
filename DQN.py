import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# ========================= Hyper Parameter ========================
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32
epoch = 10000
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
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)         
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype = torch.float, device = self.device), torch.tensor(a_lst, device = self.device), \
                torch.tensor(r_lst, device = self.device), torch.tensor(s_prime_lst, dtype=torch.float, device = self.device), \
                torch.tensor(done_mask_lst, device = self.device)
    
    def size(self):
        return len(self.buffer)

# Neural Network Model Name: Qnet
class Qnet(nn.Module):
    def __init__(self, input, output):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(input, 1240)
        self.fc2 = nn.Linear(1240, 930)
        self.fc3 = nn.Linear(930, 610)
        self.fc4 = nn.Linear(610, 320)
        self.fc5 = nn.Linear(320, output)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = torch.tensor(x, device = self.device) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
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
                for j in stock:
                    if j[0] == model:
                        if j[1] <= 0:
                            out[i] = -99999
            return out.argmax().item()
        
    def save(self, file_name = 'model.pth'):
        model_folder_path = 'DQN_save/'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()