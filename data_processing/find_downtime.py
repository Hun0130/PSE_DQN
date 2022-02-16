# -*- coding: utf-8 -*-
"""
downtime을 찾고 머신별 평균 downtime,uptime 길이와 efficiency를 계산해서 csv파일을 반환하고 저장한다.
"""
import pandas as pd
import numpy as np
import os
import glob

def load_machine_data(machine_list):
    df = []
    for i in machine_list:
        df.append(pd.read_csv(i, encoding='euc-kr'))
    return df

# 모델 디렉토리 리스트를 입력해야함. 모델별 다운타임은 딕셔너리형태로 반환
def downtime_by_model(datalist): 
    all_list = []
    for i in datalist:
        print(i)
        model = i[15:26]
        a = load_machine_data(glob.glob(os.path.join(i, "*")))
        b = find_down(a)
        model_dic = {"model" : model, "down_data" : b[0], "work_data" : b[1]}
        all_list.append(model_dic)
    return all_list

def find_down(df):    
    down_list = []
    data_list = []
    for i in df:
        down_start = []
        down_end = []
        start = []
        end = []
        for a in range(len(i.index)-1):
            if i.iloc[a,7] in ["운전중","작업대기","작업완료"] and i.iloc[a+1,7] not in ["운전중","작업대기","작업완료"]:
                down_start.append(pd.Timestamp(i.iloc[a+1,0]))
            elif i.iloc[a,7] not in ["운전중","작업대기","작업완료"] and i.iloc[a+1,7] in ["운전중","작업대기","작업완료"]:
                down_end.append(pd.Timestamp(i.iloc[a,0]))
            if a == 0 :
                start.append(pd.Timestamp(i.iloc[a,0]))
            if pd.Timestamp(i.iloc[a+1,0]) - pd.Timestamp(i.iloc[a,0]) > pd.Timedelta(minutes = 5) and a < len(i.index)-2:
                end.append(pd.Timestamp(i.iloc[a,0]))
                start.append(pd.Timestamp(i.iloc[a+1,0]))
            if a == len(i.index)-1:
                end.append(pd.Timestamp(i.iloc[a+1,0]))
        while len(down_start) != len(down_end):
            if len(down_start) > len(down_end):
                down_end.append(pd.Timestamp(i.iloc[len(i.index)-1,0]))
            if len(down_start) < len(down_end):
                down_start.insert(0, pd.Timestamp(i.iloc[0,0]))
        while len(start) != len(end):
            if len(start) > len(end):
                end.append(pd.Timestamp(i.iloc[len(i.index)-1,0]))
            if len(down_start) < len(down_end):
                start.insert(0, pd.Timestamp(i.iloc[0,0]))
        down_data = {"start" : down_start, "end" : down_end}
        data = {"start" : start, "end" : end}        
        down_df = pd.DataFrame(down_data)
        data_df = pd.DataFrame(data)
        down_df["time"] = down_df["end"]-down_df["start"]
        data_df["time"] = data_df["end"]-data_df["start"]
        down_df["machine"] = i.iloc[0,4]
        data_df["machine"] = i.iloc[0,4]
        down_list.append(down_df)
        data_list.append(data_df)
    return [down_list, data_list] # z_all

# 각 모델에서 머신별 엄타임과 다운타임을 구하는 함수
def cal_updown(downdf, workdf): 
    machine = []
    up = []
    down = []
    # a는 머신넘버
    for a in range(len(downdf)):
        if len(downdf[a].index) == 0:
            down.append(pd.Timedelta(seconds = 0))
            up.append(workdf[a]["time"].sum()/len(workdf[a].index))
        else:
            down.append(downdf[a]["time"].mean())
            up.append((workdf[a]["time"].sum()-downdf[a]["time"].sum())/(len(downdf[a].index)+1))
        machine.append(workdf[a].iloc[0,3])
    ave_time = {"average_uptime" : up, "average_downtime" : down, "machine" : machine}
    ave_time_df = pd.DataFrame(ave_time)
    ave_time_df["e"] = ave_time_df["average_uptime"]/(ave_time_df["average_uptime"]+ave_time_df["average_downtime"])
    return ave_time_df

# 모델별 전체 정보 갖고 있는 리스트 넣어서 모델별 머신별 ct 구함
def cal_ct(model_df): 
    fin_data = {}
    # i가 각 모델별 데이터프레임 리스트
    for i in model_df: 
        model_name = i[0].iloc[0,2]
        ct_list = []
        machine_list = []
        # a가 각 모델의 머신별 데이터프레임
        for a in i: 
            min_ct = pd.Timedelta(minutes = 100)
            start_time = pd.Timestamp(a.iloc[0,0])
            vir_ct = pd.Timedelta(seconds = a.iloc[0,9])
            # r 이 인덱스
            for r in range(len(a.index)-1):
                if  a.iloc[r,9] > a.iloc[r+1,9]:
                    ct = pd.Timestamp(a.iloc[r,0])-start_time
                    start_time = pd.Timestamp(a.iloc[r+1,0])
                    if pd.Timedelta(seconds = a.iloc[r,9]) > vir_ct:
                        vir_ct = pd.Timedelta(seconds = a.iloc[r,9])
                    if ct  < min_ct and ct > vir_ct:
                        min_ct = ct
            ct_list.append(min_ct)
            machine_list.append(a.iloc[0,4])
        ct_data = {"ct": ct_list, "machine":machine_list}
        ct_df = pd.DataFrame(ct_data)
        fin_data[model_name] = ct_df
    return fin_data        

def cal_product(model_df):
    fin_data = {}
    # i가 각 모델별 데이터프레임 리스트
    for i in model_df: 
        model_name = i[0].iloc[0,2]
        product_list = []
        machine_list = []
        # a가 각 모델의 머신별 데이터프레임
        for a in i: 
            product = 0
            for r in range(len(a.index)-1):#r 이 인덱스
                if  a.iloc[r,6] != a.iloc[r+1,6]:
                    product = product + 1
            product_list.append(product)
            machine_list.append(a.iloc[0,4])
        ct_data = {"product": product_list, "machine":machine_list}
        ct_df = pd.DataFrame(ct_data)
        fin_data[model_name] = ct_df
    return fin_data 

# datadic은 모델별 다운타임 저장되어 있는 dictationary, 
# alltimedata는 그냥 전체 시간 다운타임데이터, 유징데이터는 전체 데이터
def calculate(datadic, modeldata, eval_dir): 
    data_b = cal_ct(modeldata)
    data_c = cal_product(modeldata)
    # i는 모델
    try:
        os.mkdir(eval_dir)
    except:
        pass
    for i in datadic:
        mddir = eval_dir + "/" + i["model"]
        filename = mddir + "/" + i["model"] + ".csv"
        # 성능지표를 저장할 폴더를 만듬
        try:
            os.mkdir(mddir) 
        except:
            pass
        data_a = cal_updown(i["down_data"], i["work_data"])       
        data = pd.merge(data_a, data_b[i["model"]], on="machine")
        data = pd.merge(data, data_c[i["model"]], on="machine")
        data.to_csv(filename, index=False, encoding="euc-kr")       
    return

def process(month):
    machine_dir = "raw_data_machine_set"
    model_dir = "raw_data_model"
    cal_dir = "raw_data_evaluate"
    csv_name1 = "machine*_" + month + ".csv"
    allmachine_list = glob.glob(os.path.join(machine_dir, csv_name1))
    csv_name2 = "*" + month 
    allmodel_list = glob.glob(os.path.join(model_dir, csv_name2))
    
    # all_data = load_machine_data(allmachine_list)
    z_dif = downtime_by_model(allmodel_list)
    
    z_model = []

    for i in allmodel_list:
        z_model.append(load_machine_data(glob.glob(os.path.join(i, "*"))))

    save_dir = "raw_data_evaluation/" + month
    try:
        os.mkdir("raw_data_evaluation")
    except:
        pass
    calculate(z_dif, z_model, save_dir)                
    
process("06")
