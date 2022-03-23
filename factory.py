# -*- coding: utf-8 -*-
"""
ENV가 되는 Factory

"""
import copy
import pandas as pd
import numpy as np
import glob
import os
import sys
import random

from sympy import legendre 
from numba import jit, njit

# ========================= Hyper Parameter ========================
# 전체 생산 수량 낮춰주고 싶으면 조절
STOCK = 1
# ========================= Hyper Parameter ========================

# Data Structure
# df[] => (model명, model별 데이터)
# df[][0] => model명
# df[][1] => model별 데이터
# df[][1][0] => model별 데이터의 첫째행
# df[][1][0][2] => model별 데이터의 첫째행의 3번쨰 열

class factory(): 
    def __init__(self, product_list_with_df, time_df):
        # 제품 별 공정 시간 data을 sec 단위로 바꿔서 저장
        self.df = self.set_df(product_list_with_df)
        
        # #110 or #120  #150 or #160 Machine의 가능한 4개의 pattern들로 교체한 df 4개를 저장
        self.patterned_df = self.set_pattern_df(self.df)
        
        # Machine의 수 만큼 line_state_list에 ['E', 'T']를 추가함 (E = empty, T = timer)
        self.line = self.set_line(self.df)
        
        # Machine의 수 만큼 timer_list에 ['D', 0]를 추가해줌
        self.timer_list = self.make_timer_list(self.df)
        
        # 생산해야할 {모델: 개수}를 저장
        self.stock = self.set_stock(self.df)
        
        # Machine의 수 만큼 빈 buffer []를 만들어서 buffer에 저장
        self.buffer = self.set_buffer(product_list_with_df)
        
        # 최대 허용 buffer 수를 저장
        self.maxbuffer = [3, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 3, 2, 2, 5, sys.maxsize]
        
        # line과 buffer를 합쳐서 line_state로 저장 : [[line], [buffer]]
        # line_state[0][0] ~ [19] : Line state ['E', 'T']
        # line_state[1][0] ~ [19] : Buffer list []
        self.line_state = self.set_line_state(self.line, self.buffer)
        
        # 공장의 가동 시간을 모두 더해서 초 단위로 바꿔서 total_time_rank에 저장
        self.total_time_rank = int(self.sum_time(time_df)/pd.Timedelta(seconds = 1))
        
        self.avail_list = self.get_avail(self.df)
        
        # 현재 시간을 표기, 초기 값은 0
        self.now_time = 0
        
        # [[[모델, 패턴 번호], [M1, M2, M3 ...]]]로 저장
        self.choice = self.make_choice(self.patterned_df)
        
        # 생산 종료 시간 기록
        self.production_time_record = []
    
    # 모델의 time data들을 sec 단위로 바꿔서 저장 : pandas dataframe => np.array로
    def set_df(self, product_list_with_df):
        new_df = copy.deepcopy(product_list_with_df)
        df = []
        # 모든 Model들 loop
        for data_frame in new_df:
            for a in data_frame[1].index:
                # Average_Uptime을 sec 단위로 바꿔서 저장
                data_frame[1].iloc[a,0] = int(pd.Timedelta(data_frame[1].iloc[a,0])/pd.Timedelta(seconds = 1))
                # Average_Downtime을 sec 단위로 바꿔서 저장
                data_frame[1].iloc[a,1] = int(pd.Timedelta(data_frame[1].iloc[a,1])/pd.Timedelta(seconds = 1))
                # Ct(Cycle time)을 sec 단위로 바꿔서 저장
                data_frame[1].iloc[a,4] = int(pd.Timedelta(data_frame[1].iloc[a,4])/pd.Timedelta(seconds = 1))
            data_frame[1] = data_frame[1].sort_values(by='machine', ascending=True)
            df.append((data_frame[0], data_frame[1].to_numpy()))
        return df

    # #120(캘리브레이션-1) ⇒ 14 #130(캘리브레이션-2) ⇒ 15 ↔ #140(조작력-1) ⇒ 16  #150(조작력-2) => 17
    # 가능한 pattern들의 리스트를 만들어서 나눠서 저장
    def set_pattern_df(self, df):
        # 모든 패턴의 결과를 저장
        patterned_list = []
        # 모델 집합 A
        model_set_A = []
        # 모델 집합 B
        model_set_B = []
        
        # 모델을 두 분류로 나눔
        for model_set in df:
            # 만일 #130의 평균 cycle time이 20이상인 경우
            if model_set[1][4][4] > 20:
                # 모델 명을 저장 
                model_set_B.append(model_set[0])
            else:
                model_set_A.append(model_set[0])
        
        # 4개의 패턴에 따라 df를 나눠서 저장
        for idx in range(4):
            df_copied = copy.deepcopy(df)
            # #120과 #140 사용하는 경우
            if idx == 0:
                for model_set in df_copied:
                    if model_set[0] in model_set_A:
                        model_set[1][3][4] = 17
                        model_set[1][4][4] = 7
                        model_set[1][5][4] = 28
                        model_set[1][6][4] = 6
                    else:
                        model_set[1][3][4] = 34
                        model_set[1][4][4] = 7
                        model_set[1][5][4] = 31
                        model_set[1][6][4] = 6
            
            # #120과 #150을 사용하는 경우
            elif idx == 1:
                for model_set in df_copied:
                    if model_set[0] in model_set_A:
                        model_set[1][3][4] = 17
                        model_set[1][4][4] = 7
                        model_set[1][5][4] = 0
                        model_set[1][6][4] = 28
                    else:
                        model_set[1][3][4] = 34
                        model_set[1][4][4] = 7
                        model_set[1][5][4] = 0
                        model_set[1][6][4] = 31
            
            # #130과 #140 사용하는 경우
            elif idx == 2:
                for model_set in df_copied:
                    if model_set[0] in model_set_A:
                        model_set[1][3][4] = 0
                        model_set[1][4][4] = 16
                        model_set[1][5][4] = 28
                        model_set[1][6][4] = 6
                    else:
                        model_set[1][3][4] = 0
                        model_set[1][4][4] = 28
                        model_set[1][5][4] = 31
                        model_set[1][6][4] = 6

            # #130과 #150을 사용하는 경우
            elif idx == 3:
                for model_set in df_copied:
                    if model_set[0] in model_set_A:
                        model_set[1][3][4] = 0
                        model_set[1][4][4] = 16
                        model_set[1][5][4] = 0
                        model_set[1][6][4] = 28
                    else:
                        model_set[1][3][4] = 0
                        model_set[1][4][4] = 28
                        model_set[1][5][4] = 0
                        model_set[1][6][4] = 31

            patterned_list.append(df_copied)
        return patterned_list
    
    # Machine의 수 만큼 line_state_list에 ['E', 'T']를 추가함 (E = empty, T = timer)
    def set_line(self, df):
        line_state_list = []
        for i in range(len(df[0][1])):
            line_state_list.append(['E', 'T'])
        return line_state_list

    # Machine의 수 만큼 timer_list에 ['D', 0]를 추가해줌
    def make_timer_list(self, df):
        timer_list = []
        for i in range(len(df[0][1])):
            timer_list.append(['D', 0])
        return timer_list

    # 생산해야할 {모델: 개수}를 저장
    def set_stock(self, df):
        stock_state_list = dict()
        for model_set in df:
            stock_state_list[model_set[0]] = (model_set[1][19][5] // STOCK)
        return stock_state_list
    
    # Machine의 수 만큼 빈 buffer []를 만들어서 저장
    def set_buffer(self, df):
        buffer = []
        for i in range(len(df[0][1])):
            buffer.append([])
        return buffer
    
    # line과 buffer를 합쳐서 저장 : [[line], [buffer]]
    def set_line_state(self, line, buffer):
        line_state = []
        line_state.append(line)
        line_state.append(buffer)
        return line_state
    
    # 공장 raw 데이터의 가동 시간을 모두 더함
    def sum_time(self, time_df):
        total_time = pd.Timedelta(seconds = 0)
        for i in range(len(time_df.index)):
            # 공장의 가동 시간을 모두 더함
            total_time += time_df.iloc[i,2]
        return total_time
    
    # 평균 가용률 구함
    def get_avail(self, df):
        avail_list = []
        for model_set in df:
            model_list =[]
            for machine_idx in range(len(model_set[1])):
                model_list.append(model_set[1][machine_idx][3])
            avail_list.append(model_list)
        return avail_list
    
    # [[[모델, 패턴 번호], [M1, M2, M3 ...]]]을 반환
    def make_choice(self, pattern_df):
        result = []
        # 각 패턴 마다
        for i in range(len(pattern_df)):
            # 각 모델 마다
            for machine_set in pattern_df[i]:
                act = [machine_set[0], i]
                cycle_time = []
                # 각 machine의 cycle time 저장
                machine_order = [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ,1, 2, 3, 4, 5, 6, 7, 8]
                for idx in machine_order:
                    cycle_time.append(machine_set[1][idx][4])
                result.append([act, cycle_time])
        return result

    # df에서 model의 인덱스를 찾음
    def find_model_index(self, model, df):
        for i in range(len(df)):
            if df[i][0] == model:
                return i

    # 학습에 사용될 state를 생성
    def state_maker(self, line_state, time_state, pattern_df, max_buffer): 
        # 결과를 저장할 list
        state = []
        # Machine state 확인
        idx = 0
        # time_state index 계산에 사용
        time_idx = 0
        # 각 machine을 확인
        for machine_idx in line_state[idx]:
            # machine state is empty 0 * 20, 0
            if machine_idx[0] == 'E':
                # cycle time of model is 0 * 20
                for buffer_idx in range(20):
                    state.append(0)
                # time is 0
                state.append(0)

            # machine state is not empty : cycle time * 20, time
            else:
                # "empty" 대신 [model, pattern]이 저장되어 있음
                model = machine_idx[0][0]
                pattern = machine_idx[0][1]
                # "timer"에 적힌 시간을 저장
                time = machine_idx[1]
                # model, pattern에 해당하는 df에서 machine별 cycle time을 찾아서 저장
                model_idx = self.find_model_index(model, pattern_df[pattern])
                for buffer_idx in range(len(pattern_df[pattern][model_idx][1])):
                    state.append(pattern_df[pattern][model_idx][1][buffer_idx][4])
                # time을 저장
                state.append(time)
                
            # up 혹은 down 정보 추가
            if time_state[time_idx][0] == 'U':
                state.append(1)
            elif time_state[time_idx][0] == 'D':
                state.append(0)
            time_idx += 1

        # Buffer 확인
        idx = 1
        # 각 machine의 buffer를 확인
        for machine_idx in range(len(line_state[idx]) - 1):
            for buffer_idx in range(max_buffer[machine_idx]):
                # 버퍼에 제품이 존재할 시, 제품 정보를 result에 입력
                if buffer_idx < len(line_state[idx][machine_idx]):
                    model = line_state[idx][machine_idx][buffer_idx][0]
                    pattern = line_state[idx][machine_idx][buffer_idx][1]
                    # model, pattern에 해당하는 df에서 machine별 cycle time을 찾아서 저장
                    model_idx = self.find_model_index(model, pattern_df[pattern])
                    for d in range(len(pattern_df[pattern][model_idx][1])):
                        state.append(pattern_df[pattern][model_idx][1][d][4])
                # 버퍼에 제품이 없을 시, cycle time이 0이라는 의미
                else: 
                    for d in range(20):
                        state.append(0)
        return state
    
    # Machine의 UP <-> DOWN을 바꿔줌
    def make_timer(self, df, machine, model, now_state):
        model_index = self.find_model_index(model, df)
        if now_state == 'U':
            # Down 시킴
            return round(np.random.exponential(df[model_index][1][machine][1]))
        elif now_state == 'D':
            # Up 시킴
            return round(np.random.exponential(df[model_index][1][machine][0]))
    
    # 남은 총 stock 수를 구함
    def total_stock(self):
        tot_stock = 0
        for stock_num in self.stock.values():
            tot_stock += stock_num
        return tot_stock  
    
    # 리워드 계산 : #모델명, 패턴번호, patterned_df, 납입 시간, 출하 시간
    def cal_reward(self, model, pattern_idx, patterned_df, in_time, out_time):
        # 제품 출하 시각 - 제품 납입 시각 - 
        return out_time - in_time - sum(self.model_vector(model, pattern_idx, patterned_df))
    
    # 리워드 계산 종속 함수 1 : 모델의 cycle time + 패턴 벡터 리스트 반환
    def model_vector(self, model, pattern_idx, patterned_df):
        model_idx = self.find_model_index(model, patterned_df[pattern_idx])
        result = []
        for machine_idx in range(len(patterned_df[pattern_idx][model_idx][1])):
            result.append(patterned_df[pattern_idx][model_idx][1][machine_idx][4])
        
        # 패턴 인덱스의 One-Hot Encoding
        # for machine_idx in range(4):
        #     if machine_idx == pattern_idx:
        #         result.append(1)
        #     else:
        #         result.append(0)
        return result
    
    # ENV의 1 STEP : Machine1이 비워질때까지 실행
    def step(self, model, pattern):
        # 모델과 패턴을 라인에 넣어줌 (없었다면 'E')
        self.line_state[0][0][0] = [model, pattern]  
        
        model_idx = self.find_model_index(model, self.df)
        # 타이머 설정 (없었다면 'T')
        self.line_state[0][0][1] = self.df[model_idx][1][0][4] 

        # 생산했으니 재고에서 제외
        self.stock[model] -= 1
        
        # Reward 값 : 초기 값 0
        reward = 0
        
        # Machine 1이 빌 때 까지
        while self.line_state[0][0][0] != 'E':
            for machine_idx in range(len(self.line_state[0])):
                # Machine이 비어 있는 경우
                if self.line_state[0][machine_idx][0] == 'E': 
                    # Machine 전 버퍼가 비어있지 않은 경우 => 전 버퍼에서 제품을 꺼내서 Machine에 저장
                    if len(self.line_state[1][machine_idx - 1]) != 0:
                        # 버퍼의 맨 앞의 제품을 poplist에 저장
                        product = self.line_state[1][machine_idx-1].pop(0)
                        # [모델, 패턴 번호]를 저장
                        self.line_state[0][machine_idx][0] = product
                        # [M1, M2, M3 ..., M20]를 저장
                        model_idx = self.find_model_index(product[0], self.df)
                        # 제품 생산에 필요한 시간 저장
                        self.line_state[0][machine_idx][1] = self.patterned_df[product[1]][model_idx][1][machine_idx][4]
                        # 제품이 들어간 시간을 저장
                        self.line_state[0][machine_idx][0].append(self.now_time)
                    
                    # Starvation 발생 (전 버퍼가 비어있어서 진행 불가능)
                    else:
                        pass
                
                # Machine의 제품의 동작이 끝난 경우
                elif self.line_state[0][machine_idx][1] <= 0:
                    # Machine 후 버퍼의 용량이 남아 있는 경우 => Machine에서 제품을 꺼내서 후 버퍼에 저장
                    if len(self.line_state[1][machine_idx]) < self.maxbuffer[machine_idx]:
                        # 마지막 Machine에서 동작이 끝난 경우: 리워드 계산
                        if machine_idx == (len(self.line_state[0]) - 1):
                            # 리워드 계산
                            reward -= self.cal_reward(self.line_state[0][machine_idx][0][0], self.line_state[0][machine_idx][0][1], 
                            self.patterned_df, self.line_state[0][machine_idx][0][2], self.now_time)
                        
                        # Machine의 제품을 꺼내서 후 버퍼에 저장
                        product = self.line_state[0][machine_idx][0]
                        # Machine 비우기
                        self.line_state[0][machine_idx] = ['E', 'T']
                        # 버퍼에 채우기
                        self.line_state[1][machine_idx].append(product)
                    
                    # Blockage 발생 (후 버퍼의 용량이 남아있지 않아서 진행 불가능)
                    else:
                        pass
                
                # Machine의 제품의 동작이 끝나지 않은 경우 + Machine상태가 UP인 경우
                elif self.timer_list[machine_idx][0] == 'U':
                    # 제품 생산 시간 1 감소
                    self.line_state[0][machine_idx][1] -= 1
                    # 타이머 시간 1 감소
                    self.timer_list[machine_idx][1] -= 1
                    
                    # 특정 확률로 model Down 발생
                    product = self.line_state[0][machine_idx][0]
                    model_idx = self.find_model_index(product[0], self.df)
                    if self.avail_list[model_idx][machine_idx] < random.random():
                        self.timer_list[machine_idx][1] = self.make_timer(self.df, machine_idx, 
                                                                        self.line_state[0][machine_idx][0][0], 'U')
                        self.timer_list[machine_idx][0] = 'D'
                
                # Machine의 제품의 동작이 끝나지 않은 경우 + Machine상태가 DOWN인 경우
                elif self.timer_list[machine_idx][0]== 'D':
                    # 타이머 시간 1 감소
                    self.timer_list[machine_idx][1] -= 1
                    # 만일 타이머 시간이 0보다 작아지면 : state를 UP으로
                    if self.timer_list[machine_idx][1] <= 0:
                        self.timer_list[machine_idx][1] = self.make_timer(self.df, machine_idx, 
                                                                        self.line_state[0][machine_idx][0][0], 'D')
                        self.timer_list[machine_idx][0] = 'U'
                        # Stock에 다시 하나 추가
                        self.stock[self.line_state[0][machine_idx][0][0]] += 1
                        # Machine 비우기
                        self.line_state[0][machine_idx] = ['E', 'T']
                        
            # 현재 시간 + 1
            self.now_time += 1

        # Machine 1이 비었다면
        if self.line_state[0][0][0] == 'E':
            # 생산 종료
            if self.total_stock() == 0:
                self.production_time_record.append(self.now_time)
                if self.total_time_rank > self.now_time:
                    self.total_time_rank = self.now_time
                # Done is True
                return np.array(self.state_maker(self.line_state, self.timer_list, self.patterned_df, 
                                                self.maxbuffer)), reward, True, {}
            else:
                # Done is False
                return np.array(self.state_maker(self.line_state, self.timer_list, self.patterned_df, 
                                                self.maxbuffer)), reward, False, {} 
    
    # ENV reset
    def reset(self):
        self.timer_list = self.make_timer_list(self.df)
        self.stock = self.set_stock(self.df)
        self.buffer = self.set_buffer(self.df)
        self.line_state = self.set_line_state(self.line, self.buffer)
        self.now_time = 0 
        return np.array(self.state_maker(self.line_state, self.timer_list, self.patterned_df, self.maxbuffer))   
