# -*- coding: utf-8 -*-
"""
main function
"""

from unicodedata import decimal
import pandas as pd
import numpy as np
import collections
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import factory
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# https://www.youtube.com/watch?v=__NYgfkUr-M&t=960s&ab_channel=%ED%8C%A1%EC%9A%94%EB%9E%A9Pang-YoLab
learning_rate = 0.05 # 원래는 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

# ReplayBuffer
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen = buffer_limit)
    
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

        return torch.tensor(s_lst, dtype=torch.float, device = device), torch.tensor(a_lst, device = device), \
                torch.tensor(r_lst, device = device), torch.tensor(s_prime_lst, dtype=torch.float, device = device), \
                torch.tensor(done_mask_lst, device = device)
    
    def size(self):
        return len(self.buffer)

# Neural Network Model Name: Qnet
class Qnet(nn.Module):
    # Model에서 사용될 module을 정의
    def __init__(self, input, output):
        super(Qnet, self).__init__()
        # modules ...
        # Linear: y=wx+b 형태의 선형 변환을 수행하는 메소드
        # 입력되는 x의 차원과 y의 차원
        self.fc1 = nn.Linear(input, 1240)
        self.fc2 = nn.Linear(1240, 930)
        self.fc3 = nn.Linear(930, 610)
        self.fc4 = nn.Linear(610, 320)
        self.fc5 = nn.Linear(320, output)

    def forward(self, x):
        # some functions ...
        # ReLU는 max(0, x)를 의미하는 함수인데, 0보다 작아지게 되면 0이 되는 특징을 가지고 있습니다.
        x = torch.tensor(x, device = device) 
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
            return random.randint(0,len(choice)-1)
        else : 
            for i in range(len(choice)):
                model = choice[i][0][0]
                for j in stock:
                    if j[0] == model:
                        if j[1] <= 0:
                            out[i] = -99999
            return out.argmax().item()
            
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
        
def save(model, file_name = 'model.pth'):
    model_folder_path = 'DQN_save/'
    file_name = os.path.join(model_folder_path, file_name)
    torch.save(model.state_dict(), file_name)

def main():
    # 공장환경 불러오기
    product_list, time_table = factory.save_eval_data("05")
    env = factory.factory(product_list, time_table)
    
    q = Qnet(len(env.reset(product_list)), len(env.choice)).to('cuda')
    print(len(env.reset(product_list)), len(env.choice))
    
    # Target network
    q_target = Qnet(len(env.reset(product_list)), len(env.choice)).to('cuda')
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr = learning_rate)
    
    high_score = 0
    model = ''
    
    result_list = []
    now_list = []
    
    for n_epi in range(10000):
        # save
        if (high_score < score):
            high_score = score
            if os.path.isfile(('DQN_save/' + model_name)):
                os.remove(('DQN_save/' + model_name))
            model_name = 'model_' + str(n_epi) + '.pth'
            save(q.model, model_name)
        # Linear annealing from 8% to 1%
        epsilon = max(0.01, 0.08 - 0.01*(n_epi / 200)) 
        s = env.reset(product_list)
        done = False
        score = 0.0 

        while not done:
            # 1 STEP
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
            s_prime, r, done, info = env.step(env.choice[a][0][0], env.choice[a][0][1])
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                now_list.append(env.now_time)
                print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
                break
    
        # print(env.total_time_rank, env.now_time)
        result_list.append(env.total_time_rank)
        
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)
            
        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
    
    plt.subplot(211)
    plt.plot(result_list)
    plt.subplot(212)
    plt.plot(now_list)
    plt.show()
    env.close()

if __name__ == '__main__':
    main()
