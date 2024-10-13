import numpy as np 
import pandas as pd
from . import Data2D_XT_DSS
from datetime import datetime

def gauge_data_marina_reader_from_csv(gauge_data_path):
    data = pd.read_csv(gauge_data_path)
    # data.columns
    datetime_stamp =  [datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') for time_str in data['datetime']]
    return data, datetime_stamp

def rfs_data_marina_reader_from_h5(filepath):
    DSSdata = Data2D_XT_DSS.Data2D()
    DSSdata.loadh5(filepath)
    return DSSdata

def event_marina_reader(filepath): 
    df = pd.read_excel(filepath, engine='openpyxl', usecols="F:F", skiprows=2)
    stage = np.array(df.values).ravel()
    df = pd.read_excel(filepath, engine='openpyxl', usecols="H:H", skiprows=2)
    depth = np.array(df.values).ravel()
    return stage, depth

def dss_data_from_npz(filepath):
    DSSdata = Data2D_XT_DSS.Data2D()
    DSSdata = DSSdata.loadnpz(filepath)
    return DSSdata