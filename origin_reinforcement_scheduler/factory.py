# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:36:24 2020

@author: KJP
"""

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os 

#판다스 출력설정
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# processing 파일 저장
def save_eval_data(month):
    eval_dir = "raw_data_evaluation/" + month + "/"
    product_list = []
    product_list_ = glob.glob(os.path.join("eval_dir", "*"))

    for i in product_list_:
        empty_list = []
        model_name = i[22:]
        file_route = i + "/"+ model_name +".csv"
        empty_list.append(i[22:])
        empty_list.append(pd.read_csv(file_route, engine = 'python'))
        product_list.append(empty_list)
    
    time_file_route = "raw_data_stop_time/stop_time_" + month + ".csv" 
    time_table = pd.read_csv(time_file_route, engine='python')
    return product_list, time_table

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
        # self.state = self.state_maker(self.line_state)
        
    def set_df(self, product_list_with_df):
        new_df = copy.deepcopy(product_list_with_df)
        print(new_df)
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
            stock_state_list.append([i[0],i[1].iloc[18,5]])         
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
    
    def step(self, action, pattern):
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
                    return np.array(self.state_maker(self.line_state, self.timer_list, self.pattern_df, self.maxbuffer)), reward, False, {}
                else:
                    return np.array(self.state_maker(self.line_state, self.timer_list, self.pattern_df, self.maxbuffer)), 0, True, {}        

    
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
        return
    
    def render(self):
        pass

simul = []
product_list, time_table = save_eval_data("05")
env = factory(product_list, time_table)
# obs, rew, done, inf = env.step('46700-H2100', 0)

# #print(len(obs))
# #print(env.step('46700-H2100', 0))
# #print(len(env.state_maker(env.line_state, env.timer_list, env.pattern_df, env.maxbuffer)))
# env.show_state(env.line_state)
# env.step('46700-H2100', 1)
# env.show_state(env.line_state)
# env.step('46700-H2100', 2)
# env.show_state(env.line_state)
# env.step('46700-H2100', 3)
# env.show_state(env.line_state)
# env.step('46700-H2100', 0)
# env.show_state(env.line_state)
# env.step('46700-H2100', 1)
# env.show_state(env.line_state)
# env.step('46700-H2100', 2)
# env.show_state(env.line_state)
# env.step('46700-H2100', 3)
# env.show_state(env.line_state)

# print(env.pattern_df[0][0][1])
# #print(env.df[0][1])
# #print(env.buffer)
# #print(env.line)
# #print(env.maxbuffer)
# print(env.stock)
# print(env.total)
# #print(env.find_model_index("46700-H8200", env.stock))
# #print(env.make_timer(env.df, 3, "46700-H8200", "up"))
# print(env.line_state)
# print(env.timer_list)
# print(len(env.choice))
# print(env.stock)
# print(env.find_model_index('46700-H2110',env.stock))
# #print(env.total_time_rank)