import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.graph_objects as go

# this file is used to handle 3D data of discrete points

class Data3D_geometry:

    def __init__(self, datapath):
        """
        Initialize the Data3D_geometry object by loading data from my npz file.
        It could handle the well geometry data, the fracture hit data and the gauge location data.
        Parameters:
        ----------
        datapath : str
            The path to the file containing the data.
        """
        file_name = os.path.basename(datapath)
        self.filename = file_name
        dataframe = np.load(datapath, allow_pickle=True)
        self.data = dataframe['data']
        self.ew = dataframe['ew']
        self.ns = dataframe['ns']
        self.tvd = dataframe['tvd']
        self.md = dataframe['md']
        
    # info 
    def print_info(self):
        """
        Print the information of the data.
        """
        print("The data is from file: ", self.filename)
        print("The data has shape: ", self.data.shape)
        print("The data has ew: ", self.ew[:10])
        print("The data has ns: ", self.ns[:10])
        print("The data has tvd: ", self.tvd[:10])
        print("Overview of data:", self.data)

    # Data processing
    def get_md(self):
        """
        Get the MD data from the 3D data.
        Returns:
        ----------
        md : np.array
            The measured depth data.
        """
        md = []
        for i in range(len(self.ew)):
            if i == 0:
                md.append()
