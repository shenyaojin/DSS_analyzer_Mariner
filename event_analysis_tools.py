import numpy as np 
import pandas as pd
from . import Data2D_XT_DSS
from datetime import datetime

def group_elements(arr):
    grouped_array = []
    current_group = []

    # Iterate through the array
    for i in range(len(arr)):
        # If it's the first element or the same as the previous one, add to the current group
        if i == 0 or arr[i] == arr[i - 1]:
            current_group.append(arr[i])
        else:
            # If it's a new element, start a new group
            grouped_array.append(current_group)
            current_group = [arr[i]]

    # Append the last group
    grouped_array.append(current_group)
    return grouped_array
    
def event_group_generator(stage, depth): 
    # give the stage information, such as [1,2,2,2,2,2,3,3,3,3,4] and depth info [12,33,44,55,66,77,88,99,111,222,333,444]
    # reutn [[12], [33,44,55,66,77,88], [...]]
    grouped_result = group_elements(stage)
    len_list = []
    for iter in range(len(grouped_result)):
        len_list.append(len(grouped_result[iter]))
    depth_group = []
    group_index = 0
    for array_len in len_list:
        depth_group.append(np.array(depth[group_index:group_index+array_len]))
        group_index = group_index + array_len
    return depth_group

