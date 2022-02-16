"""
Created on 2022.01.18
raw data의 csv파일을 월별 데이터로 모아서 raw_data_month/raw_data_[month].csv로 저장
"""
# packages
import pandas as pd
import os
import glob
from pyarrow import csv

def cuttime(df):
    return df[1:20]

# 데이터 로드
def load_month_rawdata(month):
    # 각 행의 열 원소
    column_element = ['시간', '차종', '품번', '사양']

    # 중복된 열을 구분하기 위해 새로운 열 생성
    basic_column_element = ['공정',	'공정명', '시리얼번호', '작업상태',	'판정',	'C/T','조건표번호']
    for col_num in range(1, 21):
        for name in basic_column_element:
            new_name = name + str(col_num)
            column_element.append(new_name)
        
    # 시작 일자
    start_date = int("2019" + str(month) + "01")
    # pyarrow 읽기 옵션
    read_opts = csv.ReadOptions(encoding ="euc-kr", column_names = column_element) 
    # 한 달의 모든 파일들의 리스트
    month_csv = []
    for day in range(30):
        date = start_date + day
        # 각 일자의 데이터 디렉토리
        day_data_dir = "raw_data/" + str(date) + "/PlantableLog"
        # 각 일자의 모든 파일들의 리스트
        day_file_list = glob.glob(os.path.join(day_data_dir, "*csv"))
        day_file_list.sort()
        if len(day_file_list) > 0:
            print("load file at ", day_data_dir)
        # 각 일자의 csv 파일
        day_csv = []
        # 각 시간의 파일들을 day_csv에 저장
        for file in day_file_list:
            df = csv.read_csv(file, read_options = read_opts).to_pandas()
            day_csv.append(df)
        month_csv.extend(day_csv)
    # 모든 pandas 파일을 합침
    data_combine = pd.concat(month_csv)
    # 중복된 열 제거
    data_combine = data_combine.drop([data_combine.index[0]])
    # 시간 깔끔하게 정리
    data_combine['시간'] = pd.to_datetime(data_combine['시간'].apply(cuttime))
    result_file_name = "raw_data_" + month + ".csv"
    data_combine.to_csv(result_file_name, index = False, encoding = "euc-kr")

# 실행 파일
load_month_rawdata("05")
load_month_rawdata("06")
load_month_rawdata("07")
load_month_rawdata("08")
load_month_rawdata("09")
load_month_rawdata("10")