import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# this file is used to handle 3D data of discrete points

class Data3D_geometry:

    def __init__(self, datapath):
        """
        Initialize the Data3D_geometry object by loading data from my npz file.

        Parameters:
        ----------
        datapath : str
            The path to the file containing the data.
        """
        file_name = os.path.basename(datapath)
        self.filename = file_name
        data_structure = np.load(datapath, allow_pickle=True)