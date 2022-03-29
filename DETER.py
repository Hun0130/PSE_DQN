# -*- coding: utf-8 -*-
"""
Comparing performance of deterministic methods
"""
import random
# ==================== Choosing Model ====================        

# choose_model = "Rigid, AAABBB"
def rigid_AAAABBBB_model(choice, stock, pattern_number, model_A, model_B):
    # choose model
    run_out_check = 0
    model = 0
    for model_tuple in stock.items():
        # Model A
        if (model_tuple[1] != 0) and model_tuple[0] in model_A:
            model = model_tuple[0]
            run_out_check = 1
            break

    if run_out_check == 0:
        for model_tuple in stock.items():
            # Model B
            if (model_tuple[1] != 0) and model_tuple[0] in model_B:
                model = model_tuple[0]
                break
    
    # get choice idx
    choice_idx = 0
    for model_pattern_tuple in choice:
        if model_pattern_tuple[0][0] == model:
            if model_pattern_tuple[0][1] == pattern_number:
                return choice_idx, run_out_check
        choice_idx += 1
    return -1, run_out_check

# choose_model = "Rigid, AABBAABB"
def rigid_AABBAABB_model(choice, stock, pattern_number, idx, model_A, model_B):
    # choose model
    model = 0
    list_A = [0, 1, 4, 5]
    list_B = [2, 3, 6, 7]
    if idx in list_A:
        # Model A
        for model_tuple in stock.items():
            if (model_tuple[1] != 0) and model_tuple[0] in model_A:
                model = model_tuple[0]
                break
                
    elif idx in list_B:
        # Model B
        for model_tuple in stock.items():
            if (model_tuple[1] != 0) and model_tuple[0] in model_B:
                model = model_tuple[0]
                break        

    # get choice idx
    choice_idx = 0
    for model_pattern_tuple in choice:
        if model_pattern_tuple[0][0] == model:
            if model_pattern_tuple[0][1] == pattern_number:
                return choice_idx
        choice_idx += 1
    return -1

# choose_model = "Circular, AAAABBBB"
def circular_AAAABBBB_model(choice, stock, pattern_number, prev_model, model_A, model_B):
    # choose model
    prev_model = choice[prev_model][0][0]
    circular_check = 0
    model = 0
    run_out_check = 0
    # Model A
    for model_tuple in stock.items():
        if circular_check == 1:
            if (model_tuple[1] != 0) and model_tuple[0] in model_A:
                model = model_tuple[0]
                run_out_check = 1
                break
        if model_tuple[0] == prev_model:
            circular_check = 1
    
    # For circular
    if model == 0:
        for model_tuple in stock.items():
            # if number of model is 0
            if model_tuple[1] != 0 and model_tuple[0] in model_A:
                model = model_tuple[0]
                run_out_check = 1
                break
    
    # Model B
    if run_out_check == 0:
        for model_tuple in stock.items():
            if circular_check == 1:
                # Model B
                if (model_tuple[1] != 0) and model_tuple[0] in model_B:
                    model = model_tuple[0]
                    break
            if model_tuple[0] == prev_model:
                circular_check = 1
                
        if model == 0:
            for model_tuple in stock.items():
                # if number of model is 0
                if model_tuple[1] != 0 and model_tuple[0] in model_B:
                    model = model_tuple[0]
                    break
    
    # get choice idx
    choice_idx = 0
    for model_pattern_tuple in choice:
        if model_pattern_tuple[0][0] == model:
            if model_pattern_tuple[0][1] == pattern_number:
                return choice_idx, run_out_check
        choice_idx += 1
    return -1, run_out_check

# choose_model = "Circular, AABBAABB"
def circular_AABBAABB_model(choice, stock, pattern_number, idx, prev_model, model_A, model_B):
    # choose model
    prev_model = choice[prev_model][0][0]
    circular_check = 0
    model = 0
    list_A = [0, 1, 4, 5]
    list_B = [2, 3, 6, 7]
    
    # Model A
    if idx in list_A:
        for model_tuple in stock.items():
            if circular_check == 1:
                if (model_tuple[1] != 0) and model_tuple[0] in model_A:
                    model = model_tuple[0]
                    break
            if model_tuple[0] == prev_model:
                circular_check = 1
    
        # For circular
        if model == 0:
            for model_tuple in stock.items():
                if model_tuple[1] != 0 and model_tuple[0] in model_A:
                    model = model_tuple[0]
                    break
    
    # Model B
    elif idx in list_B:
        for model_tuple in stock.items():
            if circular_check == 1:
                if (model_tuple[1] != 0) and model_tuple[0] in model_B:
                    model = model_tuple[0]
                    break
            if model_tuple[0] == prev_model:
                circular_check = 1
        
        # For circular
        if model == 0:
            for model_tuple in stock.items():
                if model_tuple[1] != 0 and model_tuple[0] in model_B:
                    model = model_tuple[0]
                    break
    
    # get choice idx
    choice_idx = 0
    for model_pattern_tuple in choice:
        if model_pattern_tuple[0][0] == model:
            if model_pattern_tuple[0][1] == pattern_number:
                return choice_idx
        choice_idx += 1
    return -1

# choose_model = "Random, AAAABBBB"
def random_AAAABBBB_model(choice, stock, pattern_number, model_A, model_B):
    model = 0
    possible_stock = []
    run_out_check = 0
    for model_tuple in stock.items():
        if model_tuple[1] != 0 and model_tuple[0] in model_A:
            possible_stock.append(model_tuple[0])
            run_out_check = 1
            
    if len(possible_stock) == 0:
        for model_tuple in stock.items():
                if model_tuple[1] != 0 and model_tuple[0] in model_B:
                    possible_stock.append(model_tuple[0])
    
    if possible_stock:
        model = random.choice(possible_stock)
    
    # get choice idx
    choice_idx = 0
    for model_pattern_tuple in choice:
        if model_pattern_tuple[0][0] == model:
            if model_pattern_tuple[0][1] == pattern_number:
                return choice_idx, run_out_check
        choice_idx += 1
    return -1, run_out_check

# choose_model = "Random, AABBAABB" # 2 3 5(1) is best
def random_AABBAABB_model(choice, stock, pattern_number, idx, model_A, model_B):
    model = 0
    list_A = [0, 1, 4, 5]
    list_B = [2, 3, 6, 7]
    if idx in list_A:
        possible_stock = []
        for model_tuple in stock.items():
            if  model_tuple[1] != 0 and model_tuple[0] in model_A:
                possible_stock.append(model_tuple[0])
                        
    elif idx in list_B:
        possible_stock = []
        for model_tuple in stock.items():
            if model_tuple[1] != 0 and model_tuple[0] in model_B:
                possible_stock.append(model_tuple[0])
    
    if possible_stock:
        model = random.choice(possible_stock)
    
    # get choice idx
    choice_idx = 0
    for model_pattern_tuple in choice:
        if model_pattern_tuple[0][0] == model:
            if model_pattern_tuple[0][1] == pattern_number:
                return choice_idx
        choice_idx += 1
    return -1

# ==================== Choosing Model ====================

# ==================== Pattern Allocation ====================
# Pattern_num: 0(#120, #140), 1(#120, #150), 2(#130, #140), 3(#130, #150)

# choose_machine = "3-0 (Origin)"
def origin_pattern(pattern_num):
    if pattern_num == 3:
        return 0
    else:
        return 3

# choose_machine = "1-0 and 2-1 (For AAAABBBB)"
def AAAABBBB_pattern(pattern_num, check):
    if check == 0:
        if pattern_num == 2:
            return 1
        else:
            return 2
    else:
        if pattern_num == 1:
            return 0
        else:
            return 1

# choose_machine = "3-0-3-0-1-2-2-1 (For AABBAABB)"
def AABBAABB_pattern(idx):
    if idx == 0:
        return 3
    if idx == 1:
        return 0
    if idx == 2:
        return 3
    if idx == 3:
        return 0
    if idx == 4:
        return 1
    if idx == 5:
        return 2
    if idx == 6:
        return 2
    if idx == 7:
        return 1

# choose_machine = "Random Pattern" 
def random_pattern():
    return random.randrange(0, 4)
# ==================== Pattern Allocation ====================

def update_idx(prev_idx):
    if prev_idx < 7:
        now_idx = prev_idx + 1
        return now_idx
    else:
        return 0