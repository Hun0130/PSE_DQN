# -*- coding: utf-8 -*-
"""
data에서 품번 별로 데이터를 정리해주는 코드
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

def devide_data_by_model(month):
    data_dir = "raw_data_model"
    machine_dir = "raw_data_machine_set"
    csv_name = "machine*_" + month + ".csv"
    allmachine_list = glob.glob(os.path.join(machine_dir, csv_name))
    allmachine_list = sorted(allmachine_list)

    machine_data = load_machine_data(allmachine_list)
    machine_data[0]["model"] = machine_data[0]["품번"].astype("category")
    
    # model_list에 모델들 목록을 추가해줌
    model_list = machine_data[0]["model"].cat.categories 
    
    # a가 모델
    for a in model_list: 
        model_dir = data_dir + "/" + str(a) + "_" + month + "/"
        os.mkdir(model_dir)
        # i = 머신 넘버당 데이터 
        for i in machine_data: 
            model_data = i.loc[i["품번"] == a]
            model_data_dir = model_dir + str(i.iloc[0,4]) + "_" + month +".csv"
            model_data.to_csv(model_data_dir, index=False, encoding="euc-kr")
    return

devide_data_by_model("06")
