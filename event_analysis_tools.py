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

def cross_correlation(a, b):
    """
    Compute the cross-correlation of sequences a and b.

    Parameters:
    a (numpy.ndarray): First input sequence.
    b (numpy.ndarray): Second input sequence.

    Returns:
    numpy.ndarray: Cross-correlation of sequences a and b.
    """
    # or just use np.correlate(a, b, mode = 'full')
    # Ensure both sequences are 1D
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    # Length of the output sequence
    out_len = len(a) + len(b) - 1

    # Perform cross-correlation using FFT
    fft_result = np.fft.fft(a, out_len) * np.fft.fft(b[::-1], out_len)
    corr_result = np.fft.ifft(fft_result)

    return corr_result