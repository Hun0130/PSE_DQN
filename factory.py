# -*- coding: utf-8 -*-
"""
ENV가 되는 Factory Simulator
"""

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from sympy import legendre 

#판다스 출력설정
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# processing 파일 저장
def save_eval_data(month):
    eval_dir = "raw_data_evaluation/" + month + "/"
    product_list = []
    product_list_ = glob.glob(os.path.join(eval_dir, "*"))

    for i in product_list_:
        empty_list = []
        model_name = i[23:]
        file_route = i + "/"+ model_name +".csv"
        empty_list.append(i[22:])
        empty_list.append(pd.read_csv(file_route, engine = 'python'))
        product_list.append(empty_list)
    
    time_file_route = "raw_data_stop_time/stoptime_" + month + ".csv" 
    time_table = pd.read_csv(time_file_route, engine='python')
    return product_list, time_table

class factory(object): 
    def __init__(self, product_list_with_df, time_df):
        # 제품 별 공정 시간 data을 sec 단위로 바꿔서 저장
        self.df = self.set_df(product_list_with_df)
        
        # #110 or #120  #150 or #160 Machine의 가능한 4개의 pattern들로 교체한 df 4개를 pattern_df에 저장
        self.pattern_df = self.set_patter_df(self.df)
        
        # Machine의 수 만큼 line에 ['empty', 'timer']를 추가해줌
        self.line = self.set_line(self.df)
        
        # Machine의 수 만큼 timer_list에 ["down", 0]를 추가해줌
        self.timer_list = self.make_timer_list(self.df)
        
        # 생산해야할 [제품 유형, 개수]를 stock에 저장
        self.stock = self.set_stock(self.df)
        
        # Machine의 수 만큼 빈 buffer []를 만들어서 buffer에 저장
        self.buffer = self.set_buffer(product_list_with_df)
        
        # 최대 허용 buffer 수를 저장
        self.maxbuffer = [3,1,2,2,2,1,2,2,1,2,2,2,2,2,2,3,2,2,5,5000000000000]
        
        # line과 buffer를 합쳐서 line_state로 저장 : [[line], [buffer]]
        # line_state[0] = [['empty', 'timer'], ['empty', 'timer'], .... ] <machine state>
        # line_state[1] = [ [], [], [], ... ] <buffer>
        self.line_state = self.set_line_state(self.line, self.buffer)
        
        # 공장의 가동 시간을 모두 더해서 초 단위로 바꿔서 total_time_rank에 저장
        self.total_time_rank = self.sum_time(time_df)/pd.Timedelta(seconds = 1)
        
        # 현재 시간을 표기, 초기 값은 0
        self.now_time = 0
        
        # [[제품, 패턴 번호], [Machine cycle time ...]]을 choice에 저장
        self.choice = self.make_choice(self.pattern_df)
    
    # 모델의 time data들을 sec 단위로 바꿔서 저장
    def set_df(self, product_list_with_df):
        new_df = copy.deepcopy(product_list_with_df)
        # 모든 Model들 loop
        for i in new_df:
            for a in i[1].index:
                # Average_Uptime을 sec 단위로 바꿔서 저장
                i[1].iloc[a,0] = pd.Timedelta(i[1].iloc[a,0])/pd.Timedelta(seconds = 1)
                # Average_Downtime을 sec 단위로 바꿔서 저장
                i[1].iloc[a,1] = pd.Timedelta(i[1].iloc[a,1])/pd.Timedelta(seconds = 1)
                # Ct(Cycle time)을 sec 단위로 바꿔서 저장
                i[1].iloc[a,4] = pd.Timedelta(i[1].iloc[a,4])/pd.Timedelta(seconds = 1)
        return new_df

    # #110 or #120  #150 or #160 Machine의 가능한 4개의 pattern들로 교체한 df 4개를 pattern_df에 저장
    def set_patter_df(self, df):
        # #110 or #120  #150 or #160 Machine의 가능한 pattern을 저장
        pattern_list = []
        # 4개의 경우로 df를 나눠서 저장
        for i in range(4):
            new = copy.deepcopy(df)
            # #110과 #160을 사용하는 경우
            if i == 0:
                for a in new:
                    # #110 Machine의 Cycle time
                    a[1].iloc[15,4] = 9
                    # #160 Machine의 Cycle time
                    a[1].iloc[17,4] = 10
            # #110과 #150을 사용하는 경우
            elif i == 1:
                for a in new:
                    # #110 Machine의 Cycle time
                    a[1].iloc[15,4] = 10
                    # #150 Machine의 Cycle time
                    a[1].iloc[16,4] = 6 
            # #120과 #160이 사용되는 경우
            elif i == 2:
                for a in new:
                    # #120 Machine의 Cycle time
                    a[1].iloc[14,4] = 5
                    # #160 Machine의 Cycle time
                    a[1].iloc[17,4] = 9 
            # #120과 #150이 사용되는 경우
            elif i == 3:
                for a in new:
                    # #120 Machine의 Cycle time
                    a[1].iloc[14,4] = 6
                    # #150 Machine의 Cycle time
                    a[1].iloc[16,4] = 5 
            pattern_list.append(new)
        return pattern_list
    
    # Machine의 수 만큼 line_state_list에 ['empty', 'timer']를 추가해줌
    def set_line(self, df):
        line_state_list = []
        for i in df[0][1].index:
            line_state_list.append(['empty', 'timer'])
        return line_state_list

    # Machine의 수 만큼 timer_list에 ["down", 0]를 추가해줌
    def make_timer_list(self, df):
        timer_list = []
        for i in df[0][1].index:
            timer_list.append(["down", 0])
        return timer_list

    # 생산해야할 [제품 유형, 개수]를 저장
    def set_stock(self, df):
        stock_state_list = []
        for i in df:
            # 전체 생산 수량 낮춰주고 싶으면 i[1].iloc[18, 5]를 조절해주면 된다.
            stock_state_list.append([i[0],i[1].iloc[18,5]])         
        return stock_state_list
    
    # Machine의 수 만큼 빈 buffer []를 만들어서 저장
    def set_buffer(self, df):
        buffer = []
        for i in range(len(df[0][1].index)):
            buffer.append([])
        return buffer
    
    # line과 buffer를 합쳐서 저장 : [[line], [buffer]]
    def set_line_state(self, line, buffer):
        line_state = []
        line_state.append(line)
        line_state.append(buffer)
        return line_state
    
    # 공장의 가동 시간을 모두 더함
    def sum_time(self, time_df):
        total_time = pd.Timedelta(seconds = 0)
        for i in range(len(time_df.index)):
            # 공장의 가동 시간을 모두 더함
            total_time += time_df.iloc[i,2]
        return total_time
    
    # [[제품, 패턴 번호], [Machine cycle time ...]]을 반환
    def make_choice(self, pattern_df):
        result = []
        # 각 패턴 마다
        for i in range(len(pattern_df)):
            # 각 제품마다
            for a in pattern_df[i]:
                mid = []
                act = []
                ct = []
                # 제품명 저장
                act.append(a[0])
                # pattern 번호 저장 (0 1 2 3)
                act.append(i)
                for b in range(len(a[1].index)):
                    # 각 machine의 cycle time 저장
                    ct.append(a[1].iloc[b,4])
                # mid에 모두 합침
                mid.append(act)
                mid.append(ct)
                result.append(mid)
        return result
    
    # self.state_maker(self.line_state, self.timer_list, self.pattern_df, self.maxbuffer)
    # State: 머신 상태, 버퍼 상태, 머신별 작동 여부
    def state_maker(self, line_state, time_state, pattern_df, max_buffer): 
        # 결과를 저장할 list
        result = []
        
        # line_state[0] = [['empty', 'timer'], ['empty', 'timer'], .... ] <machine state>
        # line_state[1] = [ [], [], [], ... ] <buffer>
        for a in range(len(line_state)):
            # Machine state 확인
            if a == 0:
                # 각 machine을 확인
                for b in range(len(line_state[a])):
                    
                    # machine state가 empty 일 시: [0 * 20, 0]
                    if line_state[a][b][0] == "empty":
                        # machine별 cycle time이 0이라는 의미 (제품이 비어있으므로)
                        for c in range(20):
                            result.append(0)
                        # 시간이 0
                        result.append(0)
                        
                    # machine state가 empty가 아닐 시 : [cycle time * 20, time]
                    else:
                        # "empty" 대신 [model, pattern]이 저장되어 있음
                        model = line_state[a][b][0][0]
                        pattern = line_state[a][b][0][1]
                        # "timer"에 적힌 시간을 저장
                        time = line_state[a][b][1]
                        # model, pattern에 해당하는 df에서 machine별 cycle time을 찾아서 저장
                        for c in range(len(pattern_df[pattern][self.find_model_index(model, pattern_df[pattern])][1].index)):
                            result.append(pattern_df[pattern][self.find_model_index(model, pattern_df[pattern])][1].iloc[c,4])
                        # 시간을 저장
                        result.append(time)
                        
                    # 마지막에 up 혹은 down 정보 추가
                    if time_state[b][0] == "up":
                        result.append(1)
                    elif time_state[b][0] == "down":
                        result.append(0)

            # Buffer 확인
            elif a == 1:
                # 각 machine의 buffer를 확인
                for b in range(len(line_state[a])-1):
                    for c in range(max_buffer[b]):
                        # 버퍼에 제품이 존재할 시, 제품 정보를 result에 입력 : [model, pattern]이 저장되어 있음
                        if c < len(line_state[a][b]):
                            model = line_state[a][b][c][0]
                            pattern = line_state[a][b][c][1]
                            # model, pattern에 해당하는 df에서 machine별 cycle time을 찾아서 저장
                            for d in range(len(pattern_df[pattern][self.find_model_index(model, pattern_df[pattern])][1].index)):
                                result.append(pattern_df[pattern][self.find_model_index(model, pattern_df[pattern])][1].iloc[d,4])
                        # 버퍼에 제품이 없을 시, cycle time이 0이라는 의미
                        else: 
                            for d in range(20):
                                result.append(0)
        return result
    
    # df에서 model의 인덱스를 찾음
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
        self.line_state[0][0][0] = [action, pattern]  # 모델과 패턴을 라인에 넣어줌
        self.line_state[0][0][1] = self.df[self.find_model_index(action, self.df)][1].iloc[0,4] # 타이머 설정
        self.stock[self.find_model_index(action,self.stock)][1] -= 1 # 생산했으니 재고에서 제외
        
        while self.line_state[0][0][0] != "empty": # 머신 비었거나 타이머 끝나면 넘기거나 뽑아오는 거 
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
    
    def show_state(self, state):
        for i in range(len(state[0])):
            print(i,"번째 머신 : ", state[0][i][0], " timer : ",state[0][i][1],"(s)")
            if i < (len(state[0])-1):
                print(i,"번째 버퍼 : ", state[1][i])
            print("=================================================================")
        print("현재시각 : ", self.now_time)
        print("=================================================================")
        return
    
    def reset(self, df):
        self.timer_list = self.make_timer_list(self.df)
        self.stock = self.set_stock(self.df)
        self.buffer = self.set_buffer(df)
        self.line_state = self.set_line_state(self.line,self.buffer)
        self.now_time = 0
        self.choice = self.make_choice(self.pattern_df) 
        return np.array(self.state_maker(self.line_state, self.timer_list, self.pattern_df, self.maxbuffer))
    
    def render(self):
        pass