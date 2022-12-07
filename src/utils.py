from email.policy import strict
from threading import local
import torch
import copy
import re

def average_weights(w, train_img_size):
    '''
    Returns the average of the weights
    w: list of model weights of each client
    train_img_size: list of image size for training stage
    '''
    total_img_size = 0
    for img in train_img_size:
        total_img_size += img # calculate total image size for division
    w_avg = copy.deepcopy(w[0]) # get the first weight as base
    # multiply the num of image in each dataset with its weights 
    for key in w_avg.keys(): # base calculation
        w_avg[key] = w_avg[key] * train_img_size[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * train_img_size[i]
        # weighted average
        w_avg[key] = torch.div(w_avg[key], total_img_size)  
    #print(w_avg) 
    return w_avg

def selective_aggregation(specific_weight, generalized_weight, local_aggregated_model):
    '''
    Using normalization layers of generalized model, otherwise specific model.
    '''
    specific_weight = {k:v for k, v in specific_weight.items() if not re.findall("bn", k)}
    generalized_weight = {k:v for k, v in generalized_weight.items() if re.findall("bn", k)}
    
    local_aggregated_model.load_state_dict(specific_weight, strict = False)
    local_aggregated_model.load_state_dict(generalized_weight, strict = False)
    

