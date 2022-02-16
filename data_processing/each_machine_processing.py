# -*- coding: utf-8 -*-
"""
workingtime.py에서 뽑아낸 전체 작업시간을 바탕으로 
(workingtime.py에서 뽑아낸 시간은 마지막 머신 기준이므로)
각각의 머신별 작업시간을 정확하게 뽑아내서 그 작동시간을 바탕으로
각 머신의 작업데이터를 뽑아내서 저장하는 프로그램
"""
import pandas as pd
from pyarrow import csv

# csv 파일을 불러와서 반환하는 함수
def loadcsvData(file): 
    read_opts = csv.ReadOptions(encoding ="euc-kr") 
    df = csv.read_csv(file, read_options = read_opts).to_pandas()
    return df

# i번쨰 머신의 a번째 진짜 스타트를 찾는 함수
def find_machine_start(workdf, timedf, i, a):
    machinedata = workdf[i]
    x = (machinedata[machinedata['시간'] == timedf.iloc[a,0]].index)[0]
    starttime = timedf.iloc[0,0]
    for b in range(x,-1,-1):
        if b == 1:
            for c in range(x,x+600):
                if machinedata.iloc[c,9] != machinedata.iloc[c+1,9]:
                    starttime = machinedata.iloc[c+1, 0]
                    return starttime
            break        
        if machinedata.iloc[b,9] != machinedata.iloc[b-1,9]:
            starttime = machinedata.iloc[b, 0]
            return starttime
    return starttime

# i번쨰 머신의 a번쨰 진짜 피니쉬를 찾는 함수
def find_machine_finish(workdf, timedf, i, a):
    machinedata = workdf[i]
    x = (machinedata[machinedata['시간'] == timedf.iloc[a,1]].index)[0]
    for b in range(x, -1, -1):
        if b == len(machinedata.index):
            finishtime = machinedata.iloc[b, 0]
            break
        elif machinedata.iloc[b,9] != machinedata.iloc[b-1,9]:
            finishtime = machinedata.iloc[b, 0]
            break
    return finishtime 

# month raw data 파일과 month의 stop time 정보를 불러옴
def machin_processing(month):
    month_file = "raw_data_month/raw_data_" + month + ".csv" 
    rawdata = loadcsvData(month_file)
    
    time_file = "raw_data_stop_time/stoptime_" + month + ".csv"
    timedata = loadcsvData(time_file)
    rawdata['시간'] = pd.to_datetime(rawdata['시간'])
    
    # 각 머신별로 잘라서 data frame 만들어서 각 머신의 정보를 닮은 데이터프레임을 담은 리스트로 반환함
    machine_list = []
    commondf = rawdata.iloc[:, 0:4]
    for i in range((len(rawdata.columns) - 4)//7):
        machinedf = rawdata.iloc[:, 7 * i + 4 : 7 * i + 11]
        machine_list.append(pd.concat([commondf,machinedf], axis=1))
        
    test = machine_list[0].loc[(find_machine_start(machine_list, timedata, 0, 1)<=machine_list[0]["시간"]) & 
                        (machine_list[0]["시간"]<=find_machine_finish(machine_list, timedata, 0, 1)),:]

    new_splitdf = []
    new_split = []

    for i in range(len(machine_list)):
        new_splitdf.append([])    
        for a in range(len(timedata.index)):
            new_splitdf[-1].append(machine_list[i].loc[(find_machine_start(machine_list, timedata, i, a)<=machine_list[i]["시간"]) 
                                                    & (machine_list[i]["시간"]<=find_machine_finish(machine_list, timedata, i, a)),:])
        new_split.append(pd.concat(new_splitdf[-1], axis=0, ignore_index=True)) 
        
    for i in range(len(new_split)):
        if i < 10:
            number = "0" + str(i)
        else:
            number = str(i)
        machine_dir = "raw_data_machine_set" + "/machine_"+ number + "_" + month + ".csv"
        new_split[i].to_csv(machine_dir, index = False, encoding = "euc-kr")
        
machin_processing("06")