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

import factory
import DQN

def main():
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
    
    # Terminal Print
    print_interval = 20
    score = 0.0  
    
    # Save Model
    high_score = -sys.maxsize - 1
    model_name = ''
    
    reward_list = []
    production_time_list = []
    
    for n_epi in range(DQN.epoch):
        # Linear annealing from 100% to 1%
        epsilon = max(0.01, 1.0 - (1 / DQN.epoch) * n_epi) 
        s = env.reset()
        done = False
        score = 0.0 

        while not done:
            # 1 STEP
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, env.choice, env.stock)
            s_prime, reward, done = env.step(env.choice[a][0][0], env.choice[a][0][1])
            done_mask = 0.0 if done else 1.0
            
            # Memorize
            memory.put((s, a, reward/100.0, s_prime, done_mask))
            
            s = s_prime
            score += reward
            
            # End of one epoch
            if done:
                production_time_list.append(env.now_time)
                print("Episode :{}, Score : {:.1f}, High Score : {}, EPS : {:.1f}%".format(
                                                            n_epi, env.now_time, env.total_time_rank, epsilon * 100))
                break
        
        # Train
        if memory.size() > 2000:
            DQN.train(q, q_target, memory, optimizer)
        
        # Update target Q network
        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("Episode :{}, Score : {:.1f}, High Score : {}, EPS : {:.1f}%".format(
                                                            n_epi, env.now_time, env.total_time_rank, epsilon * 100))

        # Save model
        if (high_score < score):
            high_score = score
            if os.path.isfile(('DQN_save/' + model_name)):
                os.remove(('DQN_save/' + model_name))
            model_name = 'model_' + str(n_epi) + '.pth'
            q.save(model_name)
    
    plt.subplot(211)
    plt.plot(reward_list)
    plt.subplot(212)
    plt.plot(production_time_list)
    plt.show()
    env.close()

if __name__ == '__main__':
    main()
