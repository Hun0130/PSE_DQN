from calendar import month
from pyarrow import csv
import pandas
import matplotlib.pyplot as plt
import numpy as np
import random

# 시리얼 번호 7, 작업상태 8, 조건표 번호 11
def column_divide(month):
    month_data_file = "raw_data_month/raw_data_" + month + ".csv"
    # CSV read option
    read_opts = csv.ReadOptions(encoding ="euc-kr")
    df = csv.read_csv(month_data_file, read_options = read_opts).to_pandas() 
    
    serial_idx = []
    state_idx = []
    condition_idx = []
    for idx in range(1, 21):
        # 시리얼번호 
        serial_idx.append(7 * idx - 1)
        state_idx.append(7 * idx)
        condition_idx.append(7 * idx + 3)
    
    serial = df.iloc[:, serial_idx]
    state = df.iloc[:, state_idx]
    condition = df.iloc[:, condition_idx]
    
    serial_state = pandas.concat([serial, state], axis=1)
    
    result_file_name1 = "seiral_number" + month + ".csv"
    result_file_name2 = "state" + month + ".csv"
    result_file_name3 = "pattern_number" + month + ".csv"
    
    result_file_name4 = "serial_state" + month + ".csv"
    
    serial.to_csv(result_file_name1, index = False, encoding = "euc-kr")
    state.to_csv(result_file_name2, index = False, encoding = "euc-kr")
    condition.to_csv(result_file_name3, index = False, encoding = "euc-kr")
    serial_state.to_csv(result_file_name4, index = False, encoding = "euc-kr")


def visualize_state():
    state_data = "state06.csv"
    # CSV read option
    read_opts = csv.ReadOptions(encoding ="euc-kr")
    df = csv.read_csv(state_data, read_options = read_opts).to_pandas() 
    df = df.drop(df.index[0]) 
    df.columns = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15',
                    'M16', 'M17', 'M18', 'M19', 'M20']
    # df.iloc[2500:4000, :].plot()
    M = []
    for i in  range(21):
        M.append([0])
    for row in df.itertuples():
        for idx in range(1, 21):
            if row[idx] == "작업대기":
                M[idx].append(0)
            if row[idx] == "운전중":
                M[idx].append(1)
            if row[idx] == "작업완료":
                M[idx].append(2)
                
    time = []
    for i in range(12000,20000):
        time.append(i)
    
    plt.plot.bar(time, M[17][12000:20000])
    # plt.plot(time, M[18][12000:20000])
    plt.legend('best')
    plt.show()
    # df.plot()
    # plt.show()
    
def visualize_serial():
    serial_data = "serial_state06.csv"
    read_opts = csv.ReadOptions(encoding ="euc-kr")
    df = csv.read_csv(serial_data, read_options = read_opts).to_pandas() 
    df.columns = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15',
                    'M16', 'M17', 'M18', 'M19', 'M20', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 
                    'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
    
    # Save Stay time of each product
    Product_m1 = [[0,0]]
    Product_m2 = [[0,0]]
    Product_m3 = [[0,0]]
    Product_m4 = [[0,0]]
    
    # 작업대기 + 운전중 + 작업완료 시간 합
    # for row in df.iloc[:, 14:18].itertuples():
    #     # M15
    #     if row[1] != Product_m1[-1][0]:
    #         Product_m1.append([row[1], 1])
    #     if row[1] == Product_m1[-1][0]:
    #         Product_m1[-1][1] = Product_m1[-1][1] + 1
    #     # M16
    #     if row[2] != Product_m2[-1][0]:
    #         Product_m2.append([row[2], 1])
    #     if row[2] == Product_m2[-1][0]:
    #         Product_m2[-1][1] = Product_m2[-1][1] + 1
    #     # M17
    #     if row[3] != Product_m3[-1][0]:
    #         Product_m3.append([row[3], 1])
    #     if row[3] == Product_m3[-1][0]:
    #         Product_m3[-1][1] = Product_m3[-1][1] + 1
    #     # M18
    #     if row[4] != Product_m4[-1][0]:
    #         Product_m4.append([row[4], 1])
    #     if row[4] == Product_m4[-1][0]:
    #         Product_m4[-1][1] = Product_m4[-1][1] + 1
    
    # 운전중 시간만
    for row in df.iloc[:, 14:38].itertuples():
        # M15
        if(row[21] == "운전중"):
            if row[1] != Product_m1[-1][0]:
                Product_m1.append([row[1], 1])
            if row[1] == Product_m1[-1][0]:
                Product_m1[-1][1] = Product_m1[-1][1] + 1
        # M16
        if(row[22] == "운전중"):
            if row[2] != Product_m2[-1][0]:
                Product_m2.append([row[2], 1])
            if row[2] == Product_m2[-1][0]:
                Product_m2[-1][1] = Product_m2[-1][1] + 1
        # M17
        if(row[23] == "운전중"):
            if row[3] != Product_m3[-1][0]:
                Product_m3.append([row[3], 1])
            if row[3] == Product_m3[-1][0]:
                Product_m3[-1][1] = Product_m3[-1][1] + 1
        # M18
        if(row[24] == "운전중"):
            if row[4] != Product_m4[-1][0]:
                Product_m4.append([row[4], 1])
            if row[4] == Product_m4[-1][0]:
                Product_m4[-1][1] = Product_m4[-1][1] + 1
    
    while True:
        start = 2300
        end = 2500
        Product_m1 = np.array(Product_m1)[start:end]
        Product_m1 = Product_m1[Product_m1[:, 0].argsort()]
        Product_m2 = np.array(Product_m2)[start:end]
        Product_m2 = Product_m2[Product_m2[:, 0].argsort()]
        Product_m3 = np.array(Product_m3)[start:end]
        Product_m3 = Product_m3[Product_m3[:, 0].argsort()]
        Product_m4 = np.array(Product_m4)[start:end]
        Product_m4 = Product_m4[Product_m4[:, 0].argsort()]
        
        min_x = max([Product_m1[0][0], Product_m2[0][0], Product_m3[0][0], Product_m4[0][0]])
        max_x = min([Product_m1[-1][0], Product_m2[-1][0], Product_m3[-1][0], Product_m4[-1][0]])
        
        for i in range(len(Product_m1)):
            if i != len(Product_m1) - 1:
                if Product_m1[i][0] + 1 != Product_m1[i + 1][0]:
                    Product_m1 = np.append(Product_m1, np.array([[Product_m1[i][0] + 1, 0]]), axis = 0)
        
        for i in range(len(Product_m2)):
            if i != len(Product_m2) - 1:
                if Product_m2[i][0] + 1 != Product_m2[i + 1][0]:
                    Product_m2 = np.append(Product_m2, np.array([[Product_m2[i][0] + 1, 0]]), axis = 0)
                    
        for i in range(len(Product_m3)):
            if i != len(Product_m3) - 1:
                if Product_m3[i][0] + 1 != Product_m3[i + 1][0]:
                    Product_m3 = np.append(Product_m3, np.array([[Product_m3[i][0] + 1, 0]]), axis = 0)
                    
        for i in range(len(Product_m4)):
            if i != len(Product_m4) - 1:
                if Product_m4[i][0] + 1 != Product_m4[i + 1][0]:
                    Product_m4 = np.append(Product_m4, np.array([[Product_m4[i][0] + 1, 0]]), axis = 0)
        
        Product_m1 = Product_m1[Product_m1[:, 0].argsort()]
        Product_m2 = Product_m2[Product_m2[:, 0].argsort()]
        Product_m3 = Product_m3[Product_m3[:, 0].argsort()]
        Product_m4 = Product_m4[Product_m4[:, 0].argsort()]

        min_1 = 0
        max_1 = 0
        
        min_2 = 0
        max_2 = 0 
        
        min_3 = 0
        max_3 = 0
        
        min_4 = 0
        max_4 = 0
        
        for i in range(len(Product_m1)):
            if Product_m1[i][0] == min_x:
                min_1 = i
            if Product_m1[i][0] == max_x:
                max_1 = i                 
                
        for i in range(len(Product_m2)):
            if Product_m2[i][0] == min_x:
                min_2 = i
            if Product_m2[i][0] == max_x:
                max_2 = i   
        
        for i in range(len(Product_m3)):
            if Product_m3[i][0] == min_x:
                min_3 = i
            if Product_m3[i][0] == max_x:
                max_3 = i   
                
        for i in range(len(Product_m4)):
            if Product_m4[i][0] == min_x:
                min_4 = i
            if Product_m4[i][0] == max_x:
                max_4 = i   
        
        Product_m1 = np.array(Product_m1)[min_1:max_1]
        Product_m2 = np.array(Product_m2)[min_2:max_2]
        Product_m3 = np.array(Product_m3)[min_3:max_3]
        Product_m4 = np.array(Product_m4)[min_4:max_4]
            
        pr_m1 = []
        nr_m1 = []
        
        pr_m2 = []
        nr_m2 = []
        
        pr_m3 = []
        nr_m3 = []
        
        pr_m4 = []
        nr_m4 = []
        
        for element in Product_m1:
            pr_m1.append(element[0])
            nr_m1.append(element[1])
            
        for element in Product_m2:
            pr_m2.append(element[0])
            nr_m2.append(element[1])
            
        for element in Product_m3:
            pr_m3.append(element[0])
            nr_m3.append(element[1])
        
        for element in Product_m4:
            pr_m4.append(element[0])
            nr_m4.append(element[1])
            
        # for i in range(len(Product_m1)):
        #     print(i, ":", pr_m1[i], pr_m2[i], pr_m3[i], pr_m4[i])
        #     print(i, ":", nr_m1[i], nr_m2[i], nr_m3[i], nr_m4[i])
        #     input()

        nn_m2 = []
        for i in range(len(nr_m3)):
            try:
                nn_m2.append(nr_m2[i] + nr_m1[i])
            except:
                nn_m2.append(nr_m3[i])
            
        nn_m3 = []
        for i in range(len(nr_m4)):
            try:
                nn_m3.append(nr_m3[i] + nr_m2[i] + nr_m1[i])
            except:
                nn_m3.append(nr_m4[i])

        try:
            plt.bar(pr_m1, nr_m1)
            plt.bar(pr_m2, nr_m2, bottom = nr_m1)
            plt.bar(pr_m3, nr_m3, bottom = nn_m2)
            plt.bar(pr_m4, nr_m4, bottom = nn_m3)
            plt.xlabel('Product Serial')
            plt.ylabel('Production Time (t)')
            legend = plt.legend(['M15', 'M16', 'M17', 'M18'], title = "Machine")
            legend._legend_box.sep = 20
            # df.iloc[680:700, 16].plot.bar(stacked = True)
            plt.show()
            break
        except:
            continue
        
    
def visualize_pattern():
    pattern_data = "pattern_number06.csv"
    read_opts = csv.ReadOptions(encoding ="euc-kr")
    df = csv.read_csv(pattern_data, read_options = read_opts).to_pandas() 
    df.columns = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15',
                    'M16', 'M17', 'M18', 'M19', 'M20']
    df['M20'].plot()
    plt.show()

# column_divide("06")
visualize_serial()