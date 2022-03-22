# -*- coding: utf-8 -*-
"""
Comparing performance of deterministic methods
"""

import factory
import random
import main
import torch
from matplotlib import pyplot as plt

"""
env.choice: [[['model', pattern], [cycle time * 20]], [['model', pattern], [cycle time * 20]], [['model', pattern], [cycle time * 20]] ...]
    Model: env.choice[i][0][0]
    Pattern: env.choice[i][0][1]
    Cycletime: env.choice[i][1]
    
env.stock: [['Model', number], ...]
    Model: env.stock[j][0]
    Number: env.stock[j][1]
"""

# =====================================
# Choosing Model

# Rigid Method: AAABBBCCCDDD …
def rigid_model(choice, stock, pattern_number):
    for model_tuple in stock:
        # if number of model is 0
        if model_tuple[1] != 0:
            model = model_tuple[0]
            break
    
    choice_idx = 0
    for model_pattern_tuple in choice:
        if model_pattern_tuple[0][0] == model:
            if model_pattern_tuple[0][1] == pattern_number:
                return choice_idx
        choice_idx += 1
    return -1

# Circular Method: ABCDABCDABCD …
def circular_model(choice, stock, pattern_number, prev_idx):
    prev_model = choice[prev_idx][0][0]
    circular_check = 0
    model = 0
    for model_tuple in stock:
        if circular_check == 1:
            if model_tuple[1] != 0:
                model = model_tuple[0]
                break
        if model_tuple[0] == prev_model:
            circular_check = 1
            
    if model == 0:
        for model_tuple in stock:
            # if number of model is 0
            if model_tuple[1] != 0:
                model = model_tuple[0]
                break
    
    choice_idx = 0
    for model_pattern_tuple in choice:
        if model_pattern_tuple[0][0] == model:
            if model_pattern_tuple[0][1] == pattern_number:
                return choice_idx
        choice_idx += 1
    return -1

# Random Method: CBADDABCDACB …
def random_model(choice, stock, pattern_number):
    possible_stock = []
    for model_tuple in stock:
        if model_tuple[1] != 0:
            possible_stock.append(model_tuple[0])
    model = random.choice(possible_stock)
    choice_idx = 0
    for model_pattern_tuple in choice:
        if model_pattern_tuple[0][0] == model:
            if model_pattern_tuple[0][1] == pattern_number:
                return choice_idx
        choice_idx += 1
    return -1
# =====================================

# =====================================
# Pattern Allocation
# pattern_num: 0(#110, #160), 1(#110, #150), 2(#120, #160), 3(#120, #150)

# Rigid Method: 2 1 2 1 2 1 ...
def rigid_pattern(pattern_num):
    if pattern_num == 2:
        return 1
    else:
        return 2

# Circular Method: 2 1 3 0 2 1 3 0 ...
def circular_pattern_1(pattern_num):
    if pattern_num == 2:
        return 1
    if pattern_num == 3: 
        return 0
    if pattern_num == 0:
        return 2
    if pattern_num == 1:
        return 3

# Circular Method: 2 3 0 1 2 3 0 1 ...
def circular_pattern_2(pattern_num):
    if pattern_num == 2:
        return 3
    if pattern_num == 3: 
        return 0
    if pattern_num == 0:
        return 1
    if pattern_num == 1:
        return 2

# Circular Method: 2 0 3 1 2 0 3 1 ...
def circular_pattern_3(pattern_num):
    if pattern_num == 2:
        return 0
    if pattern_num == 3: 
        return 1
    if pattern_num == 0:
        return 3
    if pattern_num == 1:
        return 2

# Random Method: dacabdac ...
def random_pattern(patter_num):
    return random.randrange(0, 4)
# =====================================

# Main fuction
def single_excute():
    # 공장환경 불러오기
    product_list, time_table = factory.save_eval_data("10")
    env = factory.factory(product_list, time_table)
    s = env.reset(product_list)
    
    # stop marker
    done = False
    # pattern_num: 0(#110, #160), 1(#110, #150), 2(#120, #160), 3(#120, #150)
    pattern_num = 0
    # choice idx
    choice_idx = 0
    
    path = 'DQN_save/model_965.pth'
    # Q network
    q = main.Qnet(len(env.reset(product_list)), len(env.choice)).to('cuda')
    q.load_state_dict(torch.load(path))
    
    # main loop
    while True:
        # Choose pattern allocating method
        # pattern_num = rigid_pattern(pattern_num)
        # pattern_num = circular_pattern_1(pattern_num)
        # pattern_num = circular_pattern_2(pattern_num)
        # pattern_num = circular_pattern_3(pattern_num)
        # pattern_num = random_pattern(pattern_num)
        
        # Choose model choosing method
        # choice_idx = rigid_model(env.choice, env.stock, pattern_num)
        # choice_idx = circular_model(env.choice, env.stock, pattern_num, choice_idx)
        # choice_idx = random_model(env.choice, env.stock, pattern_num)
        
        choice_idx = q.sample_action(torch.from_numpy(s).float(), 0, env.choice, env.stock)
        
        # Error check
        if choice_idx == -1:
            print("Model Choosing Error!")
            break
        
        model = env.choice[choice_idx][0][0]
        pattern = env.choice[choice_idx][0][1]
        print("Model: ", model, " Pattern: ", pattern, " Total stock: ", env.total_stock(env.stock))
        s_prime, r, done, info = env.step(model, pattern)
        
        if env.total_stock(env.stock) == 0:
            print(env.now_time)
            break
    return env.now_time

result_time = []
for i in range(1):
    result_time.append(single_excute())
    
print(result_time)