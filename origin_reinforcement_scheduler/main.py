# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:16:40 2020

@author: KJP
"""

import glob
import os
import pandas as pd
import copy
import numpy as np
import sys

import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
#=============================================================================================================================================
product_list = []
product_list_ = glob.glob(os.path.join(r"F:\PSE\data\10th_month\proccessed_data\evaluate", "*"))

for i in product_list_:
    empty_list = []
    model_name = i[48:]
    file_route = i+r"/"+model_name+".csv"
    empty_list.append(i[48:])
    empty_list.append(pd.read_csv(file_route, engine='python'))
    product_list.append(empty_list)
   
time_table = pd.read_csv(r"F:\PSE\data\10th_month\proccessed_data\time\wholeworktime.csv", engine='python')

class factory(object): 
    def __init__(self, product_list_with_df, time_df):
        self.df = self.set_df(product_list_with_df)
        self.pattern_df = self.set_patter_df(self.df)
        self.line = self.set_line(self.df)
        self.timer_list = self.make_timer_list(self.df)
        self.stock = self.set_stock(self.df)
        self.buffer = self.set_buffer(product_list_with_df)
        self.maxbuffer = [3,1,2,2,2,1,2,2,1,2,2,2,2,2,2,3,2,2,5,5000000000000]
        self.line_state = self.set_line_state(self.line,self.buffer)
        self.total_time_rank = self.sum_time(time_df)/pd.Timedelta(seconds = 1)
        self.now_time = 0
        self.choice = self.make_choice(self.pattern_df)
        #self.state = self.state_maker(self.line_state)
        
    def set_df(self, product_list_with_df):
        new_df = copy.deepcopy(product_list_with_df)
        for i in new_df:
            for a in i[1].index:
                i[1].iloc[a,0] = pd.Timedelta(i[1].iloc[a,0])/pd.Timedelta(seconds = 1)
                i[1].iloc[a,1] = pd.Timedelta(i[1].iloc[a,1])/pd.Timedelta(seconds = 1)
                i[1].iloc[a,4] = pd.Timedelta(i[1].iloc[a,4])/pd.Timedelta(seconds = 1)
        return new_df
    
    def set_buffer(self, df):
        buffer = []
        for i in range(len(df[0][1].index)):
            buffer.append([])
        return buffer
    
    def set_patter_df(self, df):
        pattern_list = []
        for i in range(4):
            new = copy.deepcopy(df)
            if i == 0:
                for a in new:
                    a[1].iloc[15,4] = 9
                    a[1].iloc[17,4] = 10
            elif i == 1:
                for a in new:
                    a[1].iloc[15,4] = 10
                    a[1].iloc[16,4] = 6 
            elif i == 2:
                for a in new:
                    a[1].iloc[14,4] = 5
                    a[1].iloc[17,4] = 9 
            elif i == 3:
                for a in new:
                    a[1].iloc[14,4] = 6
                    a[1].iloc[16,4] = 5 
            pattern_list.append(new)
        return pattern_list
    
    def make_choice(self, pattern_df):
        result = []
        for i in range(len(pattern_df)):
            for a in pattern_df[i]:
                mid = []
                act = []
                ct = []
                act.append(a[0])
                act.append(i)
                for b in range(len(a[1].index)):
                    ct.append(a[1].iloc[b,4])
                mid.append(act)
                mid.append(ct)
                result.append(mid)
        return result
    
    def state_maker(self, line_state, time_state, pattern_df, max_buffer): # 머신 상태, 버퍼 상태, 머신별 작동 여부
        result = []
        for a in range(len(line_state)):
            if a == 0:
                for b in range(len(line_state[a])):               
                    if line_state[a][b][0] == "empty":
                        for c in range(20):
                            result.append(0)
                        result.append(0)
                    else:
                        model = line_state[a][b][0][0]
                        pattern = line_state[a][b][0][1]
                        time = line_state[a][b][1]
                        for c in range(len(pattern_df[pattern][self.find_model_index(model, pattern_df[pattern])][1].index)):
                            result.append(pattern_df[pattern][self.find_model_index(model, pattern_df[pattern])][1].iloc[c,4])
                        result.append(time)
                    if time_state[b][0] == "up":
                        result.append(1)
                    elif time_state[b][0] == "down":
                        result.append(0)
            elif a == 1:
                for b in range(len(line_state[a])-1):
                    for c in range(max_buffer[b]):
                        if c < len(line_state[a][b]): #버퍼에 있는거 넣음
                            model = line_state[a][b][c][0]
                            pattern = line_state[a][b][c][1]
                            for d in range(len(pattern_df[pattern][self.find_model_index(model, pattern_df[pattern])][1].index)):
                                result.append(pattern_df[pattern][self.find_model_index(model, pattern_df[pattern])][1].iloc[d,4])
                                
                        else: #버퍼에 머가 없음
                            for d in range(20):
                                result.append(0)
        return result
    
    def set_line(self, product_list_with_df):
        line_state_list = []
        for i in product_list_with_df[0][1].index:
            line_state_list.append(['empty', 'timer'])
        return line_state_list
        
    def set_stock(self, product_list_with_df):
        stock_state_list = []
        for i in product_list_with_df:
            stock_state_list.append([i[0],i[1].iloc[18,5]//9])       # 전체 생산 수량 낮춰주고 싶으면 이거를 나눠주면 됨
        return stock_state_list
    
    def sum_time(self, time_df):
        total_time = pd.Timedelta(seconds = 0)
        for i in range(len(time_df.index)):
            total_time += time_df.iloc[i,2]
        return total_time
    
    def set_line_state(self, line, buffer):
        line_state = []
        line_state.append(line)
        line_state.append(buffer)
        return line_state
    
    def find_model_index(self, model, df):
        for i in range(len(df)):
            if df[i][0] == model:
                return i
            
    def total_stock(self, stock):
        tot = 0
        for i in stock:
            tot += i[1]
        return tot          
    
    def model_vector(self, model, pattern, df):
        model_index = self.find_model_index(model, df[pattern])
        result = []
        for i in range(len(df[pattern][model_index][1].index)):
            result.append(df[pattern][model_index][1].iloc[i,4])
        for i in range(4):
            if i == pattern:
                result.append(1)
            else:
                result.append(0)
        return result
    
    def buffer_vector(self, maxbuf, state, df):
        result = []
        for i in range(len(maxbuf)-1):
            for a in range(maxbuf[i]):
                if a < len(state[1][i]):
                    result.extend(self.model_vector(state[1][i][a][0], state[1][i][a][1], df))
                else:
                    result.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        return result
    
    def make_input(self, state):
        result = []
        for i in range(len(state[0])):
            if state[0][i][0] == "empty":
                result.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            else:
                result.extend(self.model_vector(state[0][i][0][0], state[0][i][0][1], self.pattern_df))
                result.append(state[0][i][1])
        result.extend(self.buffer_vector(self.maxbuffer, self.line_state, self.pattern_df))
        return result
    
    def step(self, action, pattern): #액션은 품번 패턴은 15~18사이의 생산패턴
        self.line_state[0][0][0] = [action, pattern]#모델과 패턴을 라인에 넣어줌
        self.line_state[0][0][1] = self.df[self.find_model_index(action, self.df)][1].iloc[0,4]#타이머 설정
        self.stock[self.find_model_index(action,self.stock)][1] -= 1 #생산했으니 재고에서 제외
    
        while self.line_state[0][0][0] != "empty":#머신 비었거나 타이머 끝나면 넘기거나 뽑아오는 거 
            for i in range(len(self.line_state[0])):
                if self.line_state[0][i][0] == "empty": 
                    poplist = []
                    if len(self.line_state[1][i-1]) != 0:
                        poplist.append(self.line_state[1][i-1].pop(0))
                        self.line_state[0][i][0] = poplist[0]
                        self.line_state[0][i][1] = self.pattern_df[poplist[0][1]][self.find_model_index(poplist[0][0], self.df)][1].iloc[i,4]
                elif self.line_state[0][i][1] <= 0:
                    if len(self.line_state[1][i]) < self.maxbuffer[i]:
                        poplist = []
                        poplist.append(self.line_state[0][i][0])
                        self.line_state[0][i] = ["empty", 0]
                        self.line_state[1][i].append(poplist[0])
                elif self.timer_list[i][0] == "up":
                    self.line_state[0][i][1] -= 1
                    self.timer_list[i][1] -= 1
                    if self.timer_list[i][1] <= 0:
                        self.timer_list[i][1] = self.make_timer(self.df, i, self.line_state[0][i][0][0], "up")
                        self.timer_list[i][0] = "down"
                    self.now_time += 1
                elif self.timer_list[i][0]== "down":
                    self.timer_list[i][1] -= 1
                    if self.timer_list[i][1] <= 0:
                        self.timer_list[i][1] = self.make_timer(self.df, i, self.line_state[0][i][0][0], "down")
                        self.timer_list[i][0] = "up"
            if self.line_state[0][0][0] == "empty":
                stocksum = 0
                for i in self.stock:
                    stocksum += i[1]
                if stocksum == 0:
                    reward = self.total_time_rank - self.now_time
                    if self.total_time_rank > self.now_time:
                        self.total_time_rank = self.now_time
                    return np.array(self.state_maker(self.line_state, self.timer_list, self.pattern_df, self.maxbuffer)), reward, True, {}
                else:
                    return np.array(self.state_maker(self.line_state, self.timer_list, self.pattern_df, self.maxbuffer)), 0, False, {}        

    
    def make_timer(self, df, machine, model, now_state):
        model_index = self.find_model_index(model, df)
        if now_state == "up":
            return round(np.random.exponential(df[model_index][1].iloc[machine,1]))
        elif now_state == "down":
            return round(np.random.exponential(df[model_index][1].iloc[machine,0]))
    def make_timer_list(self, df):
        timer_list = []
        for i in df[0][1].index:
            timer_list.append(["down", 0])
        return timer_list
    
    def show_state(self, state):
        for i in range(len(state[0])):
            print(i,"번째 머신 : ", state[0][i][0], " timer : ",state[0][i][1],"(s)")
            if i < (len(state[0])-1):
                print(i,"번째 버퍼 : ", state[1][i])
            print("=================================================================")
        print("현재시각 : ", self.now_time)
        print("=================================================================")
        return
    
    def reset(self, product_list_with_df):
        self.timer_list = self.make_timer_list(self.df)
        self.stock = self.set_stock(self.df)
        self.buffer = self.set_buffer(product_list_with_df)
        self.line_state = self.set_line_state(self.line,self.buffer)
        self.now_time = 0
        self.choice = self.make_choice(self.pattern_df) 
        return np.array(self.state_maker(self.line_state, self.timer_list, self.pattern_df, self.maxbuffer))
     
    def render(self):
        pass
    
#=============================================================================================================================================
#https://www.youtube.com/watch?v=__NYgfkUr-M&t=960s&ab_channel=%ED%8C%A1%EC%9A%94%EB%9E%A9Pang-YoLab
learning_rate = 0.05 #원래는 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):6
        self.buffer = collections.deque(maxlen=buffer_limit)
    
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

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(1240, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 4096)
        self.fc9 = nn.Linear(4096, 4096)
        self.fc10 = nn.Linear(4096, 164)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.fc10(x)
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
            for i in range(len(out)):
                model = choice[i][0][0]
                for j in stock:
                    if j[0] == model:
                        if j[1] <= 0:
                            out[i] = -99999
            return out.argmax().item()
            #return [choice[random.randint(0,len(choice)-1)][0][0], choice[random.randint(0,len(choice)-1)][0][1]]
        #else :
            #result_list = []
            #for i in range(len(choice)):
                #dist = self.dist(out, choice[i][1])
                #result_list.append(dist)
            #return [choice[result_list.index(min(result_list))][0][0], choice[result_list.index(min(result_list))][0][1]]
            #return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        q_out = q(s)
        #print("s :",s)
        #print(q_out)
        #print(q_out.size())
        #print(a)
        #print(a.size())
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    #공장환경 불러오기
    product_list = []
    product_list_ = glob.glob(os.path.join(r"F:\PSE\data\10th_month\proccessed_data\evaluate", "*"))

    for i in product_list_:
        empty_list = []
        model_name = i[48:]
        file_route = i+r"/"+model_name+".csv"
        empty_list.append(i[48:])
        empty_list.append(pd.read_csv(file_route, engine='python'))
        product_list.append(empty_list)
   
    time_table = pd.read_csv(r"F:\PSE\data\10th_month\proccessed_data\time\wholeworktime.csv", engine='python')
    
    env = factory(product_list, time_table)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    result_list = []
    now_list = []
    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset(product_list)
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
            #print(a, env.stock[env.find_model_index(a[0],env.stock)][1])
            #print(a)
            #print(env.choice[a][0][0], env.choice[a][0][1], a)
            s_prime, r, done, info = env.step(env.choice[a][0][0], env.choice[a][0][1])
            
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime
            
            score += r
            #if env.total_stock(env.stock) <10:
                #print("거의 끝났음", env.total_stock(env.stock))
            #if r != 0:
                #print("r: ",r)
            if done:
                now_list.append(env.now_time)
                break
        print(env.total_time_rank, env.now_time)
        result_list.append(env.total_time_rank)
        
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)
            
        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    
    plt.subplot(211)
    plt.plot(result_list)
    plt.subplot(212)
    plt.plot(now_list)
    plt.show()
    env.close()

if __name__ == '__main__':
    main()
