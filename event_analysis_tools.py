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

import numpy as np

def slippage_removal(trc, detect_thres=3, data_point_removal=4, sm_N=2, abs_thres=False, local_std_N=500, is_interp=True):
    # Calculate strain rate
    trc_diff = np.diff(trc)

    # Detect slippage events
    if abs_thres:
        ind = np.abs(trc_diff) > detect_thres
    else:
        ind = np.abs(trc_diff) > detect_thres * np.std(trc_diff)
    ind = np.where(ind)[0]

    if len(ind) > len(trc) * 0.2:
        return trc

    # For each slippage, check local variance again
    if not abs_thres:
        ind_to_remove = []
        for i in range(len(ind) - 1):
            bgind = max(0, ind[i] - local_std_N // 2)
            edind = min(ind[i] + local_std_N // 2, len(trc_diff))

            local_std = np.std(trc_diff[bgind:edind])
            if np.abs(trc_diff[ind[i]]) < detect_thres * local_std:
                ind_to_remove.append(i)
        
        ind = np.delete(ind, ind_to_remove)

    # Remove data points after slippage events
    new_ind = []
    for i in ind:
        new_ind.extend(range(i, min(i + data_point_removal, len(trc_diff))))

    if len(new_ind) == 0:
        return trc

    # Perform interpolation
    good_ind = np.ones(len(trc_diff), dtype=bool)
    good_ind[new_ind] = False
    x = np.arange(len(trc_diff))
    good_trc_diff = trc_diff[good_ind].copy()

    # Apply smoothing
    if sm_N > 1:
        good_trc_diff = np.convolve(good_trc_diff, np.ones(sm_N) / sm_N, mode='same')

    # Interpolation
    if is_interp:
        trc_diff[~good_ind] = np.interp(x[~good_ind], x[good_ind], good_trc_diff)
    else:
        trc_diff[~good_ind] = 0

    # Change back to strain change
    trc_cor = np.concatenate(([trc[0]], trc[0] + np.cumsum(trc_diff)))
    return trc_cor
