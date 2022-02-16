# -*- coding: utf-8 -*-
"""
save_month_data.py에서 작업된 month 데이터의 작업시간 데이터를 보고 작업 시간을 뽑아내는 파일
raw_data_stop_time/stoptime_[month].csv와 raw_data_stop_time/day[day]_[month].csv로 저장
"""

import pandas as pd
import datetime
from pyarrow import csv
import os

# save_month_data.py에서 만들어 놓은 raw_data_[month].csv를 불러옴
def load_raw_month(month):
    # file 이름
    file = "raw_data_month/raw_data_"+ month + ".csv"
    # pyarrow 읽기 옵션
    read_opts = csv.ReadOptions(encoding ="euc-kr") 
    rawdata = csv.read_csv(file, read_options = read_opts).to_pandas()
    rawdata['시간'] = pd.to_datetime(rawdata['시간'])
    return rawdata

#전체 머신이 스탑하는 구간을 찾아서 그런 구간을 반환 해주는 함수
def findstop(data, month): 
    stoplist=[]
    startlist=[]
    onoff = 0
    # startlist.append(data.iloc[0,0])
    # stoplist.append(data.iloc[0,0])
    last_stop = data.iloc[0,0] - pd.Timedelta(minutes = 15)
    
    for a in range(len(data.index) - 1):    
        #한개라도 일하고 있는게 있는가
        working = 0 
        for i in range((len(data.columns) - 4)// 7):         
            if data.iloc[a, 9 + (7*i)] != data.iloc[a + 1, 9 + (7*i)]:         
                if onoff == 0 and i == (len(data.columns)-4)//7-1 and data.iloc[a,0]-last_stop>datetime.timedelta(minutes=10):
                    startlist.append(data.iloc[a,0])
                    onoff = 1
                working = 1 
        if (working == 0 and onoff == 1) or (onoff == 1 and data.iloc[a+1,0]-data.iloc[a,0]>datetime.timedelta(minutes=50)):
            onoff = 0
            stoplist.append(data.iloc[a, 0])
            last_stop = data.iloc[a, 0]
    
    if len(startlist) > len(stoplist):
        del startlist[-1]
    
    elif len(startlist) < len(stoplist):
        del stoplist[-1]
    
    print(len(startlist),len(stoplist))
    timetable = {'start':startlist, 'stop':stoplist}
    timetabledf = pd.DataFrame(timetable)
    timetabledf["time"] = timetabledf['stop'] - timetabledf["start"]
    timetabledf = timetabledf.loc[timetabledf["time"]>datetime.timedelta(minutes=1)]
    file_name = "raw_data_stop_time/stoptime_" + month + ".csv"
    timetabledf.to_csv(file_name, index=False, encoding="euc-kr")
    return timetabledf

# 전체 스타트스탑시간 목록을 날짜별로 정리해주는 함수, 
# 반환값은 리스트로 날짜리스트와 날짜별 시작종료시간들의 데이터그램들로 이루어져 있다. 
def dayclassify(timetable):
    daylist = []
    timelist = []
    for i in range(len(timetable.index)):
        if daylist == [] or daylist[-1] != timetable.iloc[i,0].date():
            daylist.append(timetable.iloc[i,0].date())
            timelist.append([])
            timelist[-1].append(pd.DataFrame(timetable.iloc[i,:]).transpose())
        elif daylist[-1] == timetable.iloc[i,0].date():
            timelist[-1].append(pd.DataFrame(timetable.iloc[i,:]).transpose())
    for i in range(len(timelist)):
        timelist[i] = pd.concat(timelist[i])
    return [daylist, timelist]

#data의 time폴더
def daytimefilesave(timetabledf, directory, month):
    for i in range(len(timetabledf[1])):
        filename = "/day"+str(i)+"_" + month + ".csv"
        newdir = directory + filename
        # 여기서 오류
        timetabledf[1][i].to_csv(newdir, index=False, encoding="euc-kr" )

startstop = findstop(load_raw_month("06"), "06")
daytimefilesave(dayclassify(startstop), "raw_data_stop_time", "06")

