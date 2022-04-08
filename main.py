# -*- coding: utf-8 -*-
"""
Main function
"""
from platform import machine
from turtle import Turtle
from unicodedata import decimal
import pandas as pd
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import torch
import glob
from collections import deque

import factory
import DDQN
import DETER
import DQN
import Duel_DQN
import PDQN
import AllDQN

def Deep_QN():
    # ENV setting
    product_list, time_table = factory.save_eval_data("06")
    env = factory.factory(product_list, time_table)
    
    # Q network
    q = DQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    # print(len(env.reset()), len(env.choice))
    # input()
    
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
    
    info = str(DQN.epoch) + "-" + str(DQN.train_interval) + "-" + str(DQN.learning_rate) + "-" + str(DQN.gamma) + "-" + str(env.STOCK)
    
    for n_epi in range(DQN.epoch):
        # Linear annealing from 100% to 1%
        epsilon = max(0.1, ((DQN.epoch - n_epi) / DQN.epoch)) 
        s = env.reset()
        done = False
        score = 0.0 
        step_interval = 0
        
        sa_queue = []
        sp_queue = []

        # 1 STEP : Qeueing Reward
        while not done:
            if env.total_stock() != 0:
                # select action
                a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
                # save state and action
                sa_queue.append([s, a])
                # put products
                env.put(env.choice[a][0][0], env.choice[a][0][1])
        
            # Until next action
            while True:
                reward, done, s_prime = env.step()
                done_mask = 0.0 if done else 1.0
                # Next action
                if reward == 'A':
                    sp_queue.append(s_prime)
                    break
                
                # Produce Products
                else:
                    # Memorize
                    s_r, a_r = sa_queue.pop()
                    transition = (s_r, a_r, reward, sp_queue.pop(), done_mask )
                    # transition = (s_r, a_r, reward, s_prime, done_mask )
                    score += reward
                    memory.put(transition)
                    
                if done:
                    break
                
                if env.total_stock() == 0:
                    sp_queue.append(s_prime)
            
            s = s_prime
            step_interval += 1
            
            # End of one epoch
            if done:
                production_time_list.append(env.now_time)
                reward_list.append(score)
                if len(production_time_list) > 10:
                    print("Episode :{}, Current Time : {:.1f}, Average Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".
                        format(n_epi, env.now_time, np.average(production_time_list[-10:]), 
                               env.lowest_time, score, high_score, epsilon * 100))
                else:
                    print("Episode :{}, Current Time : {:.1f}, Average Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".
                        format(n_epi, env.now_time, np.average(production_time_list), 
                               env.lowest_time, score, high_score, epsilon * 100))
                break
            
            # Train : step interval
            # if step_interval % DQN.train_interval == 0:
            #     if memory.size() > 2000:
            #         loss_list.append(DQN.train(q, memory, optimizer))
        
        # Train : 1 episode
        if memory.size() > 2000:
            loss_list.append(DQN.train_long(q, memory, optimizer))
            
        # Update target Q network
        if n_epi % DQN.update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            
        # Save model of highest reward
        if (high_score < score):
            high_score = score
            if os.path.isfile(('DQN_model/' + model_name)):
                os.remove(('DQN_model/' + model_name))
            model_name = 'model_' + str(n_epi) + '.pth'
            q.save(model_name)
        
        # Save model of highest goal
        if (env.now_time == env.lowest_time):
            if os.path.isfile(('DQN_model/' + model_name2)):
                os.remove(('DQN_model/' + model_name2))
            model_name2 = 'model2_' + str(n_epi) + '.pth'
            q.save(model_name2)
    
    file_name = 'DQN_data/' + info + "_reward" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in reward_list:
            f.write(str(i) + '\n')
            
    file_name = 'DQN_data/' + info + "_production_time" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in production_time_list:
            f.write(str(i) + '\n')

    file_name = 'DQN_data/' + info + "_loss" + '.txt'
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

def Double_DQN():
    # ENV setting
    product_list, time_table = factory.save_eval_data("06")
    env = factory.factory(product_list, time_table)
    
    # Q network
    q = DDQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    
    # Target network
    q_target = DDQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    q_target.load_state_dict(q.state_dict())
    
    # Replay Buffer
    memory = DDQN.ReplayBuffer()
    optimizer = DDQN.optim.Adam(q.parameters(), lr = DDQN.learning_rate)
    
    score = 0.0  
    
    # Save Model
    high_score = -sys.maxsize - 1
    model_name = ''
    model_name2 = ''
    
    reward_list = []
    production_time_list = []
    loss_list = []
    
    info = str(DDQN.epoch) + "-" + str(DDQN.train_interval) + "-" + str(DDQN.update_interval) + "-" + str(DDQN.learning_rate) + "-" + str(DDQN.gamma) + "-" + str(env.STOCK)
    
    for n_epi in range(DDQN.epoch):
        # Linear annealing from 100% to 1%
        epsilon = max(0.1, ((DDQN.epoch - n_epi) / DDQN.epoch)) 
        s = env.reset()
        done = False
        score = 0.0 
        step_interval = 0
        sa_queue = []
        sp_queue = []

        # 1 STEP : Queueing reward
        while not done:
            if env.total_stock() != 0:
                # select action
                a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
                # save state and action
                sa_queue.append([s, a])
                # put products
                env.put(env.choice[a][0][0], env.choice[a][0][1])
            
            # Until next action
            while True:
                reward, done, s_prime = env.step()
                done_mask = 0.0 if done else 1.0

                # Next action
                if reward == 'A':
                    sp_queue.append(s_prime)
                    break
                
                # Produce Products
                else:
                    # Memorize
                    s_r, a_r = sa_queue.pop()
                    transition = (s_r, a_r, reward, sp_queue.pop(), done_mask )
                    # transition = (s_r, a_r, reward, s_prime, done_mask )
                    score += reward
                    memory.put(transition)
                    
                if done:
                    break
                
                if env.total_stock() == 0:
                    sp_queue.append(s_prime)
            
            s = s_prime
            step_interval += 1
            
            # End of one epoch
            if done:
                production_time_list.append(env.now_time)
                reward_list.append(score)    
                if len(production_time_list) > 10:
                    print("Episode :{}, Current Time : {:.1f}, Average Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".
                        format(n_epi, env.now_time, np.average(production_time_list[-10:]), 
                               env.lowest_time, score, high_score, epsilon * 100))
                else:
                    print("Episode :{}, Current Time : {:.1f}, Average Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".
                        format(n_epi, env.now_time, np.average(production_time_list), 
                               env.lowest_time, score, high_score, epsilon * 100))
                break
        
        # Train : 1 episode
        if memory.size() > 2000:
            loss_list.append(DDQN.train_long(q, q_target, memory, optimizer))
        
        # Update target Q network
        if n_epi % DDQN.update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        # Save model of highest reward
        if (high_score < score):
            high_score = score
            if os.path.isfile(('Double_DQN_model/' + model_name)):
                os.remove(('Double_DQN_model/' + model_name))
            model_name = 'model_' + str(n_epi) + '.pth'
            q.save(model_name)
        
        # Save model of highest goal
        if (env.now_time == env.lowest_time):
            if os.path.isfile(('Double_DQN_model/' + model_name2)):
                os.remove(('Double_DQN_model/' + model_name2))
            model_name2 = 'model2_' + str(n_epi) + '.pth'
            q.save(model_name2)
    
    file_name = 'Double_DQN_data/' + info + "_reward" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in reward_list:
            f.write(str(i) + '\n')
            
    file_name = 'Double_DQN_data/' + info + "_production_time" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in production_time_list:
            f.write(str(i) + '\n')

    file_name = 'Double_DQN_data/' + info + "_loss" + '.txt'
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

def Dueling_DQN():
    # ENV setting
    product_list, time_table = factory.save_eval_data("06")
    env = factory.factory(product_list, time_table)
    
    # Q network
    q = Duel_DQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    
    # Target network
    q_target = Duel_DQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    q_target.load_state_dict(q.state_dict())
    
    # Replay Buffer
    memory = Duel_DQN.ReplayBuffer()
    optimizer = Duel_DQN.optim.Adam(q.parameters(), lr = Duel_DQN.learning_rate)
    
    score = 0.0  
    
    # Save Model
    high_score = -sys.maxsize - 1
    model_name = ''
    model_name2 = ''
    
    reward_list = []
    production_time_list = []
    loss_list = []
    
    info = str(Duel_DQN.epoch) + "-" + str(Duel_DQN.train_interval) + "-" + str(Duel_DQN.update_interval) + "-" + str(Duel_DQN.learning_rate) + "-" + str(Duel_DQN.gamma) + "-" + str(env.STOCK)
    
    for n_epi in range(Duel_DQN.epoch):
        # Linear annealing from 100% to 1%
        epsilon = max(0.1, ((Duel_DQN.epoch - n_epi) / Duel_DQN.epoch)) 
        s = env.reset()
        done = False
        score = 0.0 
        step_interval = 0

        sa_queue = []
        sp_queue = []

        # 1 STEP
        while not done:
            if env.total_stock() != 0:
                # select action
                a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
                # save state and action
                sa_queue.append([s, a])
                # put products
                env.put(env.choice[a][0][0], env.choice[a][0][1])
            
            # Until next action
            while True:
                reward, done, s_prime = env.step()
                done_mask = 0.0 if done else 1.0

                # Next action
                if reward == 'A':
                    sp_queue.append(s_prime)
                    break
                
                # Produce Products
                else:
                    # Memorize
                    s_r, a_r = sa_queue.pop()
                    transition = (s_r, a_r, reward, sp_queue.pop(), done_mask )
                    # transition = (s_r, a_r, reward, s_prime, done_mask )
                    score += reward
                    memory.put(transition)
                    
                if done:
                    break
                
                if env.total_stock() == 0:
                    sp_queue.append(s_prime)
            
            s = s_prime
            step_interval += 1
            
            # End of one epoch
            if done:
                production_time_list.append(env.now_time)
                reward_list.append(score)    
                if len(production_time_list) > 10:
                    print("Episode :{}, Current Time : {:.1f}, Average Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".
                        format(n_epi, env.now_time, np.average(production_time_list[-10:]), 
                               env.lowest_time, score, high_score, epsilon * 100))
                else:
                    print("Episode :{}, Current Time : {:.1f}, Average Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".
                        format(n_epi, env.now_time, np.average(production_time_list), 
                               env.lowest_time, score, high_score, epsilon * 100))
                break
        
        # sa_queue = []
        # sp_queue = []
        
        # # 1 STEP 
        # while not done:
        #     if env.total_stock() != 0:
        #         # select action
        #         a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
        #         # save state and action
        #         sa_queue.append([s, a])
        #         # put products
        #         env.put(env.choice[a][0][0], env.choice[a][0][1])
            
        #     reward_value = []
        #     # Until next action
        #     while True:
        #         reward, done, s_prime = env.step()
        #         done_mask = 0.0 if done else 1.0
                
        #         # Next action
        #         if reward == 'A':
        #             sp_queue.append(s_prime)
        #             if len(reward_value) != 0:
        #                 while len(sa_queue) != 0:
        #                     # Memorize
        #                     s_r, a_r = sa_queue.pop()
        #                     transition = (s_r, a_r, np.average(reward_value), sp_queue.pop(), done_mask)
        #                     score += np.average(reward_value)
        #                     memory.put(transition)
        #             break
                
        #         # Produce Products
        #         else:
        #             reward_value.append(reward)

        #         if done:
        #             break
            
        #     s = s_prime
        #     step_interval += 1
            
        #     # End of one epoch
        #     if done:
        #         production_time_list.append(env.now_time)
        #         reward_list.append(score)
        #         if len(production_time_list) > 10:
        #             print("Episode :{}, Current Time : {:.1f}, Average Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".
        #                 format(n_epi, env.now_time, np.average(production_time_list[-10:]), 
        #                        env.lowest_time, score, high_score, epsilon * 100))
        #         else:
        #             print("Episode :{}, Current Time : {:.1f}, Average Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".
        #                 format(n_epi, env.now_time, np.average(production_time_list), 
        #                        env.lowest_time, score, high_score, epsilon * 100))
        #         break
            
            # Train : step interval
            # if step_interval % Duel_DQN.train_interval == 0:
            #     if memory.size() > 2000:
            #         loss_list.append(Duel_DQN.train(q, q_target, memory, optimizer))
                    # print("Step : ", step_interval, "Loss", loss_list[-1])
        
        # # Train : 1 episode
        if memory.size() > 2000:
            loss_list.append(Duel_DQN.train_long(q, q_target, memory, optimizer))
        
        # Update target Q network
        if n_epi % Duel_DQN.update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        # Save model of highest reward
        if (high_score < score):
            high_score = score
            if os.path.isfile(('Dueling_DQN_model/' + model_name)):
                os.remove(('Dueling_DQN_model/' + model_name))
            model_name = 'model_' + str(n_epi) + '.pth'
            q.save(model_name)
        
        # Save model of highest goal
        if (env.now_time == env.lowest_time):
            if os.path.isfile(('Dueling_DQN_model/' + model_name2)):
                os.remove(('Dueling_DQN_model/' + model_name2))
            model_name2 = 'model2_' + str(n_epi) + '.pth'
            q.save(model_name2)
    
    file_name = 'Dueling_DQN_data/' + info + "_reward" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in reward_list:
            f.write(str(i) + '\n')
            
    file_name = 'Dueling_DQN_data/' + info + "_production_time" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in production_time_list:
            f.write(str(i) + '\n')

    file_name = 'Dueling_DQN_data/' + info + "_loss" + '.txt'
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

def PER_DQN():
    # ENV setting
    product_list, time_table = factory.save_eval_data("06")
    env = factory.factory(product_list, time_table)
    
    # Q network
    q = PDQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    
    # Target network
    q_target = PDQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    q_target.load_state_dict(q.state_dict())
    
    # Replay Buffer
    memory = PDQN.ReplayBuffer()
    optimizer = PDQN.optim.Adam(q.parameters(), lr = PDQN.learning_rate)
    
    score = 0.0  
    
    # Save Model
    high_score = -sys.maxsize - 1
    model_name = ''
    model_name2 = ''
    
    reward_list = []
    production_time_list = []
    loss_list = []
    
    info = str(PDQN.epoch) + "-" + str(PDQN.train_interval) + "-" + str(PDQN.update_interval) + "-" + str(PDQN.learning_rate) + "-" + str(PDQN.gamma) + "-" + str(env.STOCK)
    
    for n_epi in range(PDQN.epoch):
        # Linear annealing from 100% to 1%
        epsilon = max(0.01, ((PDQN.epoch - n_epi) / PDQN.epoch)) 
        s = env.reset()
        done = False
        score = 0.0 
        step_interval = 0

        sa_queue = []

        # 1 STEP
        while not done:
            if env.total_stock() != 0:
                # select action
                a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
                # save state and action
                sa_queue.append([s, a])
                # put products
                env.put(env.choice[a][0][0], env.choice[a][0][1])
            
            # Until next action
            while True:
                reward, done, s_prime = env.step()
                done_mask = 0.0 if done else 1.0
                # Next action
                if reward == 'A':
                    break
                
                # Produce Products
                else:
                    # Memorize
                    s_r, a_r = sa_queue.pop()
                    transition = (s_r, a_r, reward/100.0, s_prime, done_mask )
                    score += reward
                    memory.add(q, transition)
                    
                if done:
                    break
            
            s = s_prime
            step_interval += 1
            
            # End of one epoch
            if done:
                production_time_list.append(env.now_time)
                reward_list.append(score)    
                print("Episode :{}, Current Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".format(
                                                            n_epi, env.now_time, env.lowest_time, score, high_score, epsilon * 100))
                break
            
            # Train : step interval
            if step_interval % PDQN.train_interval == 0:
                if memory.size() > 2000:
                    loss_list.append(PDQN.train(q, q_target, memory, optimizer))
        
        # Train : 1 episode
        # if memory.size() > 2000:
        #     loss_list.append(PDQN.train(q, q_target, memory, optimizer))
        
        # Update target Q network
        if n_epi % PDQN.update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        # Save model of highest reward
        if (high_score < score):
            high_score = score
            if os.path.isfile(('PER_DQN_model/' + model_name)):
                os.remove(('PER_DQN_model/' + model_name))
            model_name = 'model_' + str(n_epi) + '.pth'
            q.save(model_name)
        
        # Save model of highest goal
        if (env.now_time == env.lowest_time):
            if os.path.isfile(('PER_DQN_model/' + model_name2)):
                os.remove(('PER_DQN_model/' + model_name2))
            model_name2 = 'model2_' + str(n_epi) + '.pth'
            q.save(model_name2)
    
    file_name = 'PER_DQN_data/' + info + "_reward" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in reward_list:
            f.write(str(i) + '\n')
            
    file_name = 'PER_DQN_data/' + info + "_production_time" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in production_time_list:
            f.write(str(i) + '\n')

    file_name = 'PER_DQN_data/' + info + "_loss" + '.txt'
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
    
def ALL_DQN():
    # ENV setting
    product_list, time_table = factory.save_eval_data("06")
    env = factory.factory(product_list, time_table)
    
    # Q network
    q = AllDQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    
    # Target network
    q_target = AllDQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    q_target.load_state_dict(q.state_dict())
    
    # Replay Buffer
    memory = AllDQN.ReplayBuffer()
    optimizer = AllDQN.optim.Adam(q.parameters(), lr = AllDQN.learning_rate)
    
    score = 0.0  
    
    # Save Model
    high_score = -sys.maxsize - 1
    model_name = ''
    model_name2 = ''
    
    reward_list = []
    production_time_list = []
    loss_list = []
    
    info = str(AllDQN.epoch) + "-" + str(AllDQN.train_interval) + "-" + str(AllDQN.update_interval) + "-" + str(AllDQN.learning_rate) + "-" + str(AllDQN.gamma) + "-" + str(env.STOCK)
    
    for n_epi in range(AllDQN.epoch):
        # Linear annealing from 100% to 1%
        epsilon = max(0.01, ((AllDQN.epoch - n_epi) / AllDQN.epoch)) 
        s = env.reset()
        done = False
        score = 0.0 
        step_interval = 0

        sa_queue = []

        # 1 STEP
        while not done:
            if env.total_stock() != 0:
                # select action
                a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
                # save state and action
                sa_queue.append([s, a])
                # put products
                env.put(env.choice[a][0][0], env.choice[a][0][1])
            
            # Until next action
            while True:
                reward, done, s_prime = env.step()
                done_mask = 0.0 if done else 1.0
                # Next action
                if reward == 'A':
                    break
                
                # Produce Products
                else:
                    # Memorize
                    s_r, a_r = sa_queue.pop()
                    transition = (s_r, a_r, reward/100.0, s_prime, done_mask )
                    score += reward
                    memory.add(q, transition)
                    
                if done:
                    break
            
            s = s_prime
            step_interval += 1
            
            # End of one epoch
            if done:
                production_time_list.append(env.now_time)
                reward_list.append(score)    
                print("Episode :{}, Current Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".format(
                                                            n_epi, env.now_time, env.lowest_time, score, high_score, epsilon * 100))
                break
            
            # # Train : step interval
            if step_interval % AllDQN.train_interval == 0:
                if memory.size() > 2000:
                    loss_list.append(AllDQN.train(q, q_target, memory, optimizer))
        
        # Train : 1 episode
        # if memory.size() > 2000:
        #     loss_list.append(AllDQN.train(q, q_target, memory, optimizer))
        
        # Update target Q network
        if n_epi % AllDQN.update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        # Save model of highest reward
        if (high_score < score):
            high_score = score
            if os.path.isfile(('ALL_DQN_model/' + model_name)):
                os.remove(('ALL_DQN_model/' + model_name))
            model_name = 'model_' + str(n_epi) + '.pth'
            q.save(model_name)
        
        # Save model of highest goal
        if (env.now_time == env.lowest_time):
            if os.path.isfile(('ALL_DQN_model/' + model_name2)):
                os.remove(('ALL_DQN_model/' + model_name2))
            model_name2 = 'model2_' + str(n_epi) + '.pth'
            q.save(model_name2)
    
    file_name = 'ALL_DQN_data/' + info + "_reward" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in reward_list:
            f.write(str(i) + '\n')
            
    file_name = 'ALL_DQN_data/' + info + "_production_time" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in production_time_list:
            f.write(str(i) + '\n')

    file_name = 'ALL_DQN_data/' + info + "_loss" + '.txt'
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
    
def ALL_DQN_break():
    # ENV setting
    product_list, time_table = factory.save_eval_data("06")
    env = factory.factory(product_list, time_table)
    
    # Q network
    q = AllDQN.Qnet(len(env.reset_break()), len(env.choice)).to('cuda')
    
    # Target network
    q_target = AllDQN.Qnet(len(env.reset_break()), len(env.choice)).to('cuda')
    q_target.load_state_dict(q.state_dict())
    
    # Replay Buffer
    memory = AllDQN.ReplayBuffer()
    optimizer = AllDQN.optim.Adam(q.parameters(), lr = AllDQN.learning_rate)
    
    score = 0.0  
    
    # Save Model
    high_score = -sys.maxsize - 1
    model_name = ''
    model_name2 = ''
    
    reward_list = []
    production_time_list = []
    loss_list = []
    
    info = str(AllDQN.epoch) + "-" + str(AllDQN.train_interval) + "-" + str(AllDQN.update_interval) + "-" + str(AllDQN.learning_rate) + "-" + str(AllDQN.gamma) + "-" + str(env.STOCK)
    
    for n_epi in range(AllDQN.epoch):
        # Linear annealing from 100% to 1%
        epsilon = max(0.01, ((AllDQN.epoch - n_epi) / AllDQN.epoch)) 
        s = env.reset_break()
        done = False
        score = 0.0 
        step_interval = 0

        sa_queue = []

        # 1 STEP
        while not done:
            if env.total_stock() != 0:
                # select action
                a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
                # save state and action
                sa_queue.append([s, a])
                # put products
                env.put(env.choice[a][0][0], env.choice[a][0][1])
            
            # Until next action
            while True:
                reward, done, s_prime = env.step(1)
                done_mask = 0.0 if done else 1.0
                # Next action
                if reward == 'A':
                    break
                
                # Produce Products
                else:
                    # Memorize
                    s_r, a_r = sa_queue.pop()
                    transition = (s_r, a_r, reward/100.0, s_prime, done_mask )
                    score += reward
                    memory.add(q, transition)
                    
                if done:
                    break
            
            s = s_prime
            step_interval += 1
            
            # End of one epoch
            if done:
                production_time_list.append(env.now_time)
                reward_list.append(score)    
                print("Episode :{}, Current Time : {:.1f}, Lowest Time : {}, Score: {:1f}, High Score: {:1f} EPS : {:.1f}%".format(
                                                            n_epi, env.now_time, env.lowest_time, score, high_score, epsilon * 100))
                break
            
            # Train : step interval
            if step_interval % AllDQN.train_interval == 0:
                if memory.size() > 2000:
                    loss_list.append(AllDQN.train(q, q_target, memory, optimizer))
        
        # Train : 1 episode
        # if memory.size() > 2000:
        #     loss_list.append(AllDQN.train(q, q_target, memory, optimizer))
        
        # Update target Q network
        if n_epi % AllDQN.update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        # Save model of highest reward
        if (high_score < score):
            high_score = score
            if os.path.isfile(('ALL_DQN_BREAK_model/' + model_name)):
                os.remove(('ALL_DQN_BREAK_model/' + model_name))
            model_name = 'model_' + str(n_epi) + '.pth'
            q.save_break(model_name)
        
        # Save model of highest goal
        if (env.now_time == env.lowest_time):
            if os.path.isfile(('ALL_DQN_BREAK_model/' + model_name2)):
                os.remove(('ALL_DQN_BREAK_model/' + model_name2))
            model_name2 = 'model2_' + str(n_epi) + '.pth'
            q.save_break(model_name2)
    
    file_name = 'ALL_DQN_BREAK_data/' + info + "_reward" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in reward_list:
            f.write(str(i) + '\n')
            
    file_name = 'ALL_DQN_BREAK_data/' + info + "_production_time" + '.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in production_time_list:
            f.write(str(i) + '\n')

    file_name = 'ALL_DQN_BREAK_data/' + info + "_loss" + '.txt'
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
    # Save Downtime 
    down_time = []
    # Save starvation time
    starvation_time = []
    # Save blockage tome
    blockage_time = []
    
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
            
            # put products
            if env.total_stock() > 0:
                env.put(model, pattern)
            
            # Until next action
            while True:
                reward, done, s_prime = env.step()
                # Next action
                if reward == 'A':
                    break
                
                if done:
                    break
            
            # env.show_state()
            # input()

            # End of one epoch
            if done:
                result_time.append(env.now_time)
                down_time.append(env.down_time)
                starvation_time.append(env.starvation_time)
                blockage_time.append(env.blockage_time)
                break
    
    file_name = 'Deter_data/' + choose_model + choose_machine + '_production.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in result_time:
            f.write(str(i) + '\n')
            
    file_name = 'Deter_data/' + choose_model + choose_machine +  '_down_time.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in down_time:
            f.write(str(i) + '\n')
            
    file_name = 'Deter_data/' + choose_model + choose_machine +  '_starvation_time.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in starvation_time:
            f.write(str(i) + '\n')
            
    file_name = 'Deter_data/' + choose_model + choose_machine +  '_blockage_time.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in blockage_time:
            f.write(str(i) + '\n')
    
    print("Result: ", sum(result_time) / iter_num)
    return

def Test(iter_num, folder, test_file):
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
    
    path = folder + test_file + '.pth'
    # Q network
    q = DDQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    q.load_state_dict(torch.load(path))
    print("Test file is ", test_file)
    # Save Result
    result_time = []
    # Save Downtime 
    down_time = []
    # Save starvation time
    starvation_time = []
    # Save blockage tome
    blockage_time = []
    
    # Save input log
    input_log = []

    for iter in range(iter_num):
        s = env.reset()
        done = False
        
        # 1 STEP
        while not done:
            if env.total_stock() != 0:
                # select action
                a = q.sample_action(torch.from_numpy(s).float(), 0.08, env.choice, env.stock)
                # put products
                env.put(env.choice[a][0][0], env.choice[a][0][1])
            
            # Until next action
            while True:
                reward, done, s_prime = env.step()
                done_mask = 0.0 if done else 1.0
                # Next action
                if reward == 'A':
                    break                    
                if done:
                    break
            
            s = s_prime

            # End of one epoch
            if done:
                result_time.append(env.now_time)
                down_time.append(env.down_time)
                starvation_time.append(env.starvation_time)
                blockage_time.append(env.blockage_time)
                # print(env.now_time)
                break

    file_name = 'Deter_data/' + test_file + '_production_time.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in result_time:
            f.write(str(i) + '\n')
    
    file_name = 'Deter_data/' + test_file + '_down_time.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in down_time:
            f.write(str(i) + '\n')
            
    file_name = 'Deter_data/' + test_file + '_starvation_time.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in starvation_time:
            f.write(str(i) + '\n')
            
    file_name = 'Deter_data/' + test_file + '_blockage_time.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in blockage_time:
            f.write(str(i) + '\n')
            
    
    print("Result: ", sum(result_time) / iter_num)
    return

def Analysis(folder, test_file):
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
    
    path = folder + test_file + '.pth'
    # Q network
    q = DDQN.Qnet(len(env.reset()), len(env.choice)).to('cuda')
    q.load_state_dict(torch.load(path))
    print("Test file is ", test_file)
    # Save Result
    result_time = []
    # Save Downtime 
    down_time = []
    # Save starvation time
    starvation_time = []
    # Save blockage tome
    blockage_time = []
    
    # Save input log
    input_log = []
    
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
        log = "Model: " + model + " Pattern: " + str(pattern) + " Model set: " + env.print_model(model)
        input_log.append(log)
        
        s_prime, reward, done = env.step(model, pattern)
        s = s_prime

        # End of one epoch
        if done:
            result_time.append(env.now_time)
            down_time.append(env.down_time)
            starvation_time.append(env.starvation_time)
            blockage_time.append(env.blockage_time)
            # print(env.now_time)
            break
    
    file_name = 'Deter_data/' + test_file + '_input_model.txt'
    with open(file_name,'w', encoding='UTF-8') as f:
        for i in input_log:
            f.write(str(i) + '\n')
            
    
    print("Result: ", sum(result_time), sum(blockage_time[0]), sum(starvation_time[0]))
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

    # # Raw data 뽑기
    # data_name = []
    # for file_name in file_list_:
    #     data_name.append(file_name[11:-3])
    
    # 평균내서 가공
    # data_list = []
    # data_dir = "Deter_data/"
    # # opt_list = ["*-0 3-0.txt", "*-1 3-0-3-0-1-2-2-1.txt", "*-0 1-0+2-1.txt", "* Random.txt"]
    # opt_list = ["C-0 *", "R-0 *", "Ri-0 *", "C-1 *", "R-1 *", "Ri-1 *"]
    # opt_list = ["C-0 *3-0.txt", "R-0 *3-0.txt", "Ri-0 *3-0.txt", "C-1 *3-0.txt", "R-1 *3-0.txt", "Ri-1 *3-0.txt"]
    # for opt in opt_list:
    #     file_list_ = glob.glob(os.path.join(data_dir, opt))
    #     data = []
    #     for file_name in file_list_:
    #         file = open(file_name, "r")
    #         while True:
    #             line = file.readline()
    #             if not line:
    #                 break
    #             data.append(5120 / float(line) * 60)
    #         file.close()
    #     data_list.append(data)
    
    # 평균내서 가공
    # data_name = ["3-0", "3-0-3-0-1-2-2-1", "1-0+2-1", "Random"]
    # data_name = ["C-0", "R-0", "Ri-0", "C-1", "R-1", "Ri-1"]
    
    # 다른 알고리즘과 비교
    data_list = []
    data_dir = "Deter_data/"
    opt_list = ["Ri-0 3-0.txt", "Ri-0 1-0+2-1.txt", "Ri-1 Random.txt", 
                "C-1 3-0.txt", "C-1 3-0-3-0-1-2-2-1.txt", "C-1 Random.txt",
                "R-0 3-0.txt", "R-0 1-0+2-1.txt", "R-0 Random.txt", 
                "result11_production_time.txt", "result12_production_time.txt", "result1.txt",]
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
        

    plt.style.use('default')
    fig, ax = plt.subplots()
    
    data_name = ["Rigid (Rigid)", "Rigid (Numerical Optimal)", "Rigid (Random)",
                "Circular (Rigid)", "Circular (Numerical Optimal)", "Circular (Random)",
                "Random (Rigid)", "Random (Numerical Optimal)", "Random (Random)", "DQN without target Q", 
                "Dueling DQN", "Double DQN"]
    
    ax.boxplot(data_list, notch=True)
    ax.legend(data_name, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Algorithm Type - Throughput')
    ax.set_xlabel('Algorithm Type')
    ax.set_ylabel('Throughput (products / min )')

    plt.show()

def Graph_Log():
    data_dir = "Deter_data/"
    
    block_list = []
    file_list_ = glob.glob(os.path.join(data_dir, "_blockage_time.txt"))
    for file_name in file_list_:
        data = [[] for i in range(20)]
        file = open(file_name, "r")
        while True:
            line = file.readline()
            if not line:
                break
            line = line.split(',')
            for i in range(len(line)):
                if i == 0:
                    line[i] = int(line[i][1:])
                if i == len(line) - 1:
                    line[i] = int(line[i][:len(line[i])-2])
                else:
                    line[i] = int(line[i])
            for machine_idx in range(len(line)):
                data[machine_idx].append(int(line[machine_idx]))
        for i in range(len(data)):
            data[i] = float(np.average(data[i]))
        block_list.append(data)
        file.close()
        
    starv_list = []
    file_list_ = glob.glob(os.path.join(data_dir, "_starvation_time.txt"))
    for file_name in file_list_:
        data = [[] for i in range(20)]
        file = open(file_name, "r")
        while True:
            line = file.readline()
            if not line:
                break
            line = line.split(',')
            for i in range(len(line)):
                if i == 0:
                    line[i] = int(line[i][1:])
                if i == len(line) - 1:
                    line[i] = int(line[i][:len(line[i])-2])
                else:
                    line[i] = int(line[i])
            for machine_idx in range(len(line)):
                data[machine_idx].append(int(line[machine_idx]))
        for i in range(len(data)):
            data[i] = float(np.average(data[i]))
        starv_list.append(data)
        file.close()
    print(starv_list[0])
    machine_list = np.arange(20)
    for idx in range(len(starv_list)):
        plt.bar(machine_list - 0.1*idx, starv_list[idx], label = idx, width = 0.1)
    # for graph in block_list:
    #     ax.bar(machine_list, graph, label = 'Blockage time (sec)')
    # plt.bar.set_xlabel('Machine number')
    # plt.set_ylabel('Time(sec)')
    plt.show()

if __name__ == '__main__':
    # Deep_QN()
    # Double_DQN()
    Dueling_DQN()
    
    # PER_DQN()
    # ALL_DQN()
    # ALL_DQN_break()

    # Graph_Deter()
    
    # Test
    # Deter(100, 1, 1)
    # Deter(100, 1, 2)
    # Deter(100, 2, 4)
    # Deter(100, 4, 1)
    # Deter(100, 4, 2)
    # Deter(100, 4, 4)
    # Deter(100, 5, 1)
    # Deter(100, 5, 2)
    # Deter(100, 5, 4)
    
    # Test(10, 'DQN_model/',"[220976] model_9125")
    # Test(10, 'DQN_model/',"model2_7101")
    
    #Test(100, 'Double_DQN_model/',"model_8627")
    # Test(100, 'Double_DQN_model/',"model2_8574")
    
    # Test(100, 'Dueling_DQN_model/',"model_853")
    # Test(100, 'Dueling_DQN_model/',"model2_2204")
    
    # Test(100, 'PER_DQN_model/',"model_4237")
    # Test(100, 'PER_DQN_model/',"model_9218")
    # Test(100, 'PER_DQN_model/',"model2_4237")
    # Test(100, 'PER_DQN_model/',"model2_4861")
    
    # Test(100, 'ALL_DQN_model/',"model_9376")
    # Test(100, 'ALL_DQN_model/',"model_7344")
    # Test(100, 'ALL_DQN_model/',"model2_2754")
    # Test(100, 'ALL_DQN_model/',"model2_3073")

    # Graph_Log()
    
    # Analysis('Double_DQN_model/', "result10")