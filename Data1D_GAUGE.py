import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import os
from copy import deepcopy

# define an object to store the data
class Data1D_GAUGE:
    """
    A class to represent 1D gauge data.

    Attributes:
    ----------
    filename : str
        The name of the file containing the data.
    data : numpy.ndarray
        The gauge data values.
    taxis : numpy.ndarray
        The datetime values corresponding to the data.
    """

    # constructor
    def __init__(self, filename):
        """
        Initialize the Data1D_GAUGE object by loading data from my npz file.

        Parameters:
        ----------
        filename : str
            The path to the file containing the data.
        """
        file_name = os.path.basename(filename)
        self.filename = file_name
        data_structure = np.load(filename, allow_pickle=True)
        self.data = data_structure['value']
        self.taxis = data_structure['datetime']

    def crop(self, start, end):
        """
        Crop the data to a specific time range.

        Parameters:
        ----------
        start : datetime
            The start time for cropping.
        end : datetime
            The end time for cropping.
        """
        ind = (self.taxis >= start) & (self.taxis <= end)
        self.data = self.data[ind]
        self.taxis = self.taxis[ind]

    def shift(self, shift):
        """
        Apply a time shift to the data.

        Parameters:
        ----------
        shift : timedelta
            The time shift to apply.
        """
        self.taxis = self.taxis + shift

    def plot_simple(self, use_datetime=True):
        """
        Plot the gauge data.
        """
        if use_datetime == True:
            plt.figure()
            plt.plot(self.taxis, self.data)
            # set xtick rotation
            plt.xticks(rotation=30)
            plt.title("Gauge data: " + self.filename)
            plt.xlabel("Time")
            plt.ylabel("Pressure (psi)")
            plt.tight_layout()
            plt.show()
        else:
            time_axis = self.calculate_time() * 3600
            plt.figure()
            plt.plot(time_axis, self.data)
            plt.title("Gauge data: " + self.filename)
            plt.xlabel("Time (sec)")
            plt.ylabel("Pressure (psi)")
            plt.tight_layout()
            plt.show()
    def print_info(self):
        """
        Print information about the gauge data.
        """
        print("Gauge data: " + self.filename)
        print("Number of data points: " + str(len(self.data)))
        print("Time range: " + str(self.taxis[0]) + " - " + str(self.taxis[-1]))

    def copy(self):
        """
        Return a copy of the Data1D_GAUGE object.
        """
        new_data = deepcopy(self)
        return new_data

    def get_start_time(self):
        """
        Return the start time of the data.
        """
        return self.taxis[0]
    
    def get_end_time(self):
        """
        Return the end time of the data.
        """
        return self.taxis[-1]
    
    def plot(self, ax): 
        """
        Plot the gauge data on a given axis.

        Parameters:
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the data.
        """
        ax.plot(self.taxis, self.data)
        ax.set_title("Gauge data: " + self.filename)
        ax.set_xlabel("Time")
        ax.set_ylabel("Pressure (psi)")
        ax.tick_params(axis='x', rotation=30)
        # legend
        ax.legend([self.filename], loc='upper right')

    def export(self, filename):
        """
        Export the gauge data to a CSV file.

        Parameters:
        ----------
        filename : str
            The name of the file to export the data to.
        
        WARNING: UNDER TESTING. NOT SURE IF IT WORKS.
        """
        df = pd.DataFrame({'datetime': self.taxis, 'value': self.data})
        df.to_csv(filename, index=False)

    def calculate_time(self):
        """
        Calculate the time axis in hours from the datetime axis.
        """
        self.time = np.array([(t - self.taxis[0]).total_seconds() / 3600 for t in self.taxis])

        return self.time
    
    def calculate_dqdt(self):
        """
        Calculate the rate of change of the gauge data.
        """
        
        self.calculate_time()
        self.dqdt = np.gradient(self.data)[1:] / self.time[1:]

        return self.dqdt