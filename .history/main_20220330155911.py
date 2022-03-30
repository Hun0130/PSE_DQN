# -*- coding: utf-8 -*-
"""
Main function
"""
from unicodedata import decimal
import pandas as pd
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import torch
import glob

import factory
import DQN
import DETER

def DDQN():
    # ENV setting
    product_list, time_table = factory.save_eval_data("06")
    env = factory.factory(product_list, time_table)
    
    # Q network
    q = DQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    
    # Target network
    q_target = DQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    q_target.load_state_dict(q.state_dict())
    
    # Replay Buffer
    memory = DQN.ReplayBuffer()
    optimizer = DQN.optim.Adam(q.parameters(), lr = DQN.learning_rate)
    
    score = 0.0  
    
    # Save Model
    high_score = -sys.maxsize - 1
    model_name = ''
    model_name2 = ''
    
    reward_list = []
    production_time_list = []
    loss_list = []
    
    info = str(DQN.epoch) + "-" + str(DQN.train_interval) + "-" + str(DQN.update_interval) + "-" + str(DQN.learning_rate) + "-" + str(DQN.gamma) + "-" + str(env.STOCK)
    
    for n_epi in range(DQN.epoch):
        # Linear annealing from 100% to 1%
        epsilon = max(0.01, ((DQN.epoch - n_epi) / DQN.epoch)) 
        s = env.reset()
        done = False
        score = 0.0 
        step_interval = 0

        while not done:
            # 1 STEP
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
            s_prime, reward, done = env.step(env.choice[a][0][0], env.choice[a][0][1])
            done_mask = 0.0 if done else 1.0
            
            # Memorize
            memory.put((s, a, reward/100.0, s_prime, done_mask))
            
            s = s_prime
            score += reward
            step_interval += 1
            
            # End of one epoch
            if done:
                production_time_list.append(env.now_time)
                reward_list.append(score)    
                print("Episode :{}, Current Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".format(
                                                            n_epi, env.now_time, env.lowest_time, score, high_score, epsilon * 100))
                break
            
            # # Train : step interval
            # if step_interval % DQN.train_interval == 0:
            #     if memory.size() > 2000:
            #         loss_list.append(DQN.train(q, q_target, memory, optimizer))
        
        # Train : 1 episode
        if memory.size() > 2000:
            loss_list.append(DQN.train(q, q_target, memory, optimizer))
        
        # Update target Q network
        if n_epi % DQN.update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        # Save model of highest reward
        if (high_score < score):
            high_score = score
            if os.path.isfile(('DDQN_model/' + model_name)):
                os.remove(('DDQN_model/' + model_name))
            model_name = 'model_' + str(n_epi) + '.pth'
            q.save(model_name)
        
        # Save model of highest goal
        if (env.now_time == env.lowest_time):
            if os.path.isfile(('DDQN_model/' + model_name2)):
                os.remove(('DDQN_model/' + model_name2))
            model_name2 = 'model2_' + str(n_epi) + '.pth'
            q.save(model_name2)
    
    file_name = 'DDQN_data/' + info + "_reward" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in reward_list:
            f.write(str(i) + '\n')
            
    file_name = 'DDQN_data/' + info + "_production_time" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in production_time_list:
            f.write(str(i) + '\n')

    file_name = 'DDQN_data/' + info + "_loss" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in loss_list:
            f.write(str(i) + '\n')
    
    plt.subplot(311)
    plt.plot(reward_list)
    plt.title("Reward")
    plt.subplot(312)
    plt.plot(production_time_list)
    plt.title("Production Time")
    plt.subplot(313)
    plt.plot(loss_list)
    plt.title("Loss")
    plt.show()

def Deter(iter_num, model_option, machine_option):
    # ENV setting
    product_list, time_table = factory.save_eval_data("06")
    env = factory.factory(product_list, time_table)
    env.reset()
    
    # Pattern_num: 0(#120, #140), 1(#120, #150), 2(#130, #140), 3(#130, #150)
    pattern_num = 0
    # choice idx
    choice_idx = 0

    # Option Check
    if model_option == 1:
        choose_model = "Rigid, AAAABBBB" # 1 0 (For A) 2 1 (For B)
    if model_option == 2:
        choose_model = "Rigid, AABBAABB" # 3, 0, 7, 4, 1, 2, 6, 5 => 3 0 3 0 1 2 2 1 
    if model_option == 3:
        choose_model = "Circular, AAAABBBB" # 1 0 (For A) 2 1 (For B)
    if model_option == 4:
        choose_model = "Circular, AABBAABB" # 3, 0, 7, 4, 1, 2, 6, 5 => 3 0 3 0 1 2 2 1 
    if model_option == 5:
        choose_model = "Random, AAAABBBB" # 1 0 (For A) 2 1 (For B)
    if model_option == 6:
        choose_model = "Random, AABBAABB" # 3, 0, 7, 4, 1, 2, 6, 5 => 3 0 3 0 1 2 2 1 
        
    if machine_option == 1:
        choose_machine = "3-0 (Origin)"
    if machine_option == 2:
        choose_machine = "1-0 and 2-1 (For AAAABBBB)"
    if machine_option == 3:
        choose_machine = "3-0-3-0-1-2-2-1 (For AABBAABB)"
    if machine_option == 4:
        choose_machine = "Random Pattern" 
    print("STOCK: %s Model Schedule: %s Pattern Schedule: %s" %(env.total_stock(), choose_model, choose_machine ))
    
    # Save Result
    result_time = []
    for iter in range(iter_num):
        # Reset env
        env.reset()
        # machine allocating pattern number
        pattern_num = -1
        
        # check for AAAABBBB pattern
        check = 1
        
        # for AABBAABB pattern
        prev_idx = 0
        
        # check the running out B
        run_out_check = 0
        
        # Test Loop
        while True:
            if run_out_check != 1:
                # Choose pattern allocating method
                if machine_option == 1:
                    pattern_num = DETER.origin_pattern(pattern_num)
                if machine_option == 2:
                    pattern_num = DETER.AAAABBBB_pattern(pattern_num, check)
                if machine_option == 3:
                    pattern_num = DETER.AABBAABB_pattern(prev_idx)
                if machine_option == 4:
                    pattern_num = DETER.random_pattern()
                
                if model_option == 1:
                    choice_idx, check = DETER.rigid_AAAABBBB_model(env.choice, env.stock, pattern_num, env.model_set_A, env.model_set_B)
                    prev_idx = DETER.update_idx(prev_idx)
                if model_option == 2:
                    choice_idx = DETER.rigid_AABBAABB_model(env.choice, env.stock, pattern_num, prev_idx ,env.model_set_A, env.model_set_B)
                    prev_idx = DETER.update_idx(prev_idx)
                if model_option == 3:
                    choice_idx, check = DETER.circular_AAAABBBB_model(env.choice, env.stock, pattern_num, choice_idx, 
                                                            env.model_set_A, env.model_set_B)
                    prev_idx = DETER.update_idx(prev_idx)
                if model_option == 4:
                    choice_idx = DETER.circular_AABBAABB_model(env.choice, env.stock, pattern_num, prev_idx, choice_idx, 
                                                            env.model_set_A, env.model_set_B)
                    prev_idx = DETER.update_idx(prev_idx)
                if model_option == 5:
                    choice_idx, check = DETER.random_AAAABBBB_model(env.choice, env.stock, pattern_num, env.model_set_A, env.model_set_B)
                    prev_idx = DETER.update_idx(prev_idx)
                if model_option == 6:
                    choice_idx = DETER.random_AABBAABB_model(env.choice, env.stock, pattern_num, prev_idx, env.model_set_A, env.model_set_B)
                    prev_idx = DETER.update_idx(prev_idx)
                    
            else:
                pattern_num = DETER.AAAABBBB_pattern(pattern_num, 1)
                choice_idx, check = DETER.rigid_AAAABBBB_model(env.choice, env.stock, pattern_num, env.model_set_A, env.model_set_B)
            
            if choice_idx == -1:
                run_out_check = 1
            
            model = env.choice[choice_idx][0][0]
            pattern = env.choice[choice_idx][0][1]
            s_prime, reward, done = env.step(model, pattern)
            
            # print(model, pattern, env.print_model(model))
            # input()

            # End of one epoch
            if done:
                result_time.append(env.now_time)
                break
    
    file_name = 'Deter_data/' + choose_model + choose_machine + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in result_time:
            f.write(str(i) + '\n')
    
    print("Result: ", sum(result_time) / iter_num)
    return
    
def Test(iter_num, test_file):
    # ENV setting
    product_list, time_table = factory.save_eval_data("06")
    env = factory.factory(product_list, time_table)
    env.reset()
    
    # stop marker
    done = False
    # Pattern_num: 0(#120, #140), 1(#120, #150), 2(#130, #140), 3(#130, #150)
    pattern_num = 0
    # choice idx
    choice_idx = 0
    
    path = 'DDQN_model/' + test_file + '.pth'
    # Q network
    q = DQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    q.load_state_dict(torch.load(path))
    print("Test file is ", test_file)
    # Save Result
    result_time = []
    for iter in range(iter_num):
        # Reset env
        s = env.reset()
        while True:
            # Choose pattern allocating method
            choice_idx = q.sample_action(torch.from_numpy(s).float(), 0, env.choice, env.stock)
            
            # Error check
            if choice_idx == -1:
                print("Model Choosing Error!")

            model = env.choice[choice_idx][0][0]
            pattern = env.choice[choice_idx][0][1]
            # print("Model: ", model, " Pattern: ", pattern, " Total stock: ", env.total_stock())
            
            s_prime, reward, done = env.step(model, pattern)
            s = s_prime

            # End of one epoch
            if done:
                result_time.append(env.now_time)
                # print(env.now_time)
                break

    file_name = test_file + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in result_time:
            f.write(str(i) + '\n')
    
    print("Result: ", sum(result_time) / iter_num)
    return


def Graph_Deter():
    # # Raw data 뽑기
    # data_list = []
    # data_dir = "Deter_data/"
    # file_list_ = glob.glob(os.path.join(data_dir, "*3-0.txt"))
    # for file_name in file_list_:
    #     data = []
    #     file = open(file_name, "r")
    #     while True:
    #         line = file.readline()
    #         if not line:
    #             break
    #         data.append(5120 / float(line) * 60)
    #     data_list.append(data)
    #     file.close()
    
    # 평균내서 가공
    data_list = []
    data_dir = "Deter_data/"
    opt_list = ["*-0 3-0.txt", "*-1 3-0-3-0-1-2-2-1.txt", "*-0 1-0+2-1.txt", "* Random.txt"]
    for opt in opt_list:
        file_list_ = glob.glob(os.path.join(data_dir, opt))
        data = []
        for file_name in file_list_:
            file = open(file_name, "r")
            while True:
                line = file.readline()
                if not line:
                    break
                data.append(5120 / float(line) * 60)
            file.close()
        data_list.append(data)
    
    
    data_name = []
    for file_name in file_list_:
        data_name.append(file_name[11:-3])

    plt.style.use('default')
    fig, ax = plt.subplots()
    
    ax.boxplot(data_list, notch=True)
    ax.legend(data_name, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Algorithm Type - Throughput')
    ax.set_xlabel('Algorithm Type')
    ax.set_ylabel('Throughput (products / min )')

    plt.show()

if __name__ == '__main__':
    # DDQN()
    # Deter(100, 6, 4)
    # Test(100, "result1")
    Graph_Deter()
