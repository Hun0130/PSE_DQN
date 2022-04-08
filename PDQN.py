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
buffer_limit  = 50000
batch_size    = 32
epoch = 1000
train_interval = 10
update_interval = 50
# ========================= Hyper Parameter ========================

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

# ReplayBuffer
class ReplayBuffer():
    e = 0.01
    a = 0.8
    beta = 0.3
    beta_increment_per_sampling = 0.0005
    
    def __init__(self):
        self.tree = SumTree(buffer_limit)
        self.capacity = buffer_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        
    def _get_priority(self, error):
            return ((abs(error)) + self.e) ** self.a
    
    def add(self, q, sample):
        s = torch.tensor(np.array(sample[0]), dtype = torch.float, device = self.device)
        a = torch.tensor(np.array(sample[1]), device = self.device)
        r = torch.tensor(np.array(sample[2]), dtype = torch.float, device = self.device)
        s_prime = torch.tensor(np.array(sample[3]), dtype=torch.float, device = self.device)
        done_mask = torch.tensor(np.array(sample[4]), dtype = torch.float, device = self.device)
        
        q_out = q(s)
        q_a = q(s)[sample[1]]
        max_q_prime = torch.max(q(s_prime))
        
        target = r + gamma * max_q_prime * done_mask
        error = abs(q_a - target)
        p = self._get_priority(error)
        self.tree.add(p, sample)
    
    def sample(self, n):
        s_lst = []
        a_lst = []
        r_lst = [] 
        s_prime_lst = []
        done_mask_lst = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            s, a, r, s_prime, done_mask = data
            s_lst.append(s)         
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return torch.tensor(np.array(s_lst), dtype = torch.float, device = self.device), torch.tensor(np.array(a_lst), 
                device = self.device), torch.tensor(np.array(r_lst), dtype = torch.float, device = self.device), \
                torch.tensor(np.array(s_prime_lst), dtype=torch.float, device = self.device),\
                torch.tensor(np.array(done_mask_lst), dtype = torch.float, device = self.device), idxs, is_weight
    
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    
    def size(self):
        return self.tree.total()

# Neural Network Model Name: Qnet
class Qnet(nn.Module):
    def __init__(self, input, output):
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(input, 1240)
        self.fc2 = nn.Linear(1240, 930)
        self.fc3 = nn.Linear(930, 610)
        self.fc4 = nn.Linear(610, 320)
        self.fc5 = nn.Linear(320, output)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = torch.Tensor.clone(x).detach().to('cuda')
        # x = torch.tensor(x, device = self.device) 
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
                for j in stock.items():
                    if j[0] == model:
                        if j[1] <= 0:
                            out[i] = -99999
            return out.argmax().item()
        
    def save(self, file_name = 'model.pth'):
        model_folder_path = 'PER_DQN_model/'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
def train(q, q_target, memory, optimizer):
    result = 0
    for i in range(10):
        s,a,r,s_prime,done_mask, idxs, is_weight = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        
        errors = torch.abs(q_a - target).data.cpu().numpy()
        
        # update priority
        for i in range(batch_size):
            idx = idxs[i]
            memory.update(idx, errors[i])
        
        loss = F.smooth_l1_loss(q_a, target)
        result += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return result
