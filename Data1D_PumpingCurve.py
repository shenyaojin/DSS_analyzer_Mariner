import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

class Data1D_PumpingCurve:
    """
    A class to represent 1D pumping curve data.
    """

    def __init__(self, filename):
        """
        Initialize the Data1D_PumpingCurve object by loading data from a npz file.

        Parameters:
        ----------
        filename : str
            The path to the file containing the data.
        """
        file_name = os.path.basename(filename)
        self.filename = file_name
        data_structure = np.load(filename, allow_pickle=True)
        self.label = data_structure['label']
        self.taxis = data_structure['taxis']
        self.data = data_structure['value']
    
    def gen_unit(self):
        """
        Generate the unit of the data.

        Returns:
        ----------
        str
            The unit of the data.
        """
        # create a dictionary of units
        unit_dict = {'Treating Pressure': 'psi', 'Slurry Rate': 'bpm', 'Proppant Concentration': 'lb/gal'}
        # find the variable index
        self.unit = []
        for key in self.label:
            self.unit.append(unit_dict[key])

    def print_label(self):
        """
        Print the label of the data.
        """
        print(self.label)

    def print_info(self):
        """
        Print the information of the data.
        """
        print('Filename:', self.filename)
        print('Label:', self.label)
        print('Time range:', self.taxis[0], "-", self.taxis[-1])

    def get_start_time(self):
        """
        Get the start time of the data.

        Returns:
        ----------
        datetime
            The start time of the data.
        """
        return self.taxis[0]
    
    def get_end_time(self):
        """
        Get the end time of the data.

        Returns:
        ----------
        datetime
            The end time of the data.
        """
        return self.taxis[-1]
    
    def apply_timeshift(self, shift):
        """
        Apply a time shift to the data.

        Parameters:
        ----------
        shift : timedelta
            The time shift to apply.
        """
        self.taxis = self.taxis + shift

    def plot_single_var(self, key):
        """
        Plot the data of a single variable.

        Parameters:
        ----------
        key : str
            The name of the variable to plot.
        """
        
        # find the variable index
        var_index = np.where(self.label == key)[0][0]

        # plot the variable
        plt.figure()
        plt.plot(self.taxis, self.data[var_index, :])
        plt.xlabel('Time')
        plt.ylabel(key)
        plt.title(key)
        plt.tight_layout()
        plt.show()
    
    def get_data_by_name(self, key):
        """
        Get the data of a variable by name.

        Parameters:
        ----------
        key : str
            The name of the variable to get.

        Returns:
        ----------
        np.array
            The data of the variable.
        """
        # find the variable index
        var_index = np.where(self.label == key)[0][0]
        return self.data[var_index, :]
    
    def plot_all_vars_simple(self, figsize=(10, 6)):
        """
        Plot the data of all variables.
        """

        plt.figure(figsize=figsize)
        self.gen_unit()
        # Read the pumping curve data and plot it
        ax3 = plt.subplot2grid((3,4), (2,0), colspan=4, rowspan=1)
        color = 'blue'
        ax3.set_xlabel('Time')
        ax3.set_ylabel(f'{self.label[0]}/{self.unit[0]}', color=color)
        ax3.plot(self.taxis, self.data[0,:], label=self.label[0], color=color)
        ax3.tick_params(axis='y', labelcolor=color)

        # Making a second axis for treating pressure
        ax31 = ax3.twinx()
        color = 'green'
        ax31.set_ylabel(f'{self.label[1]}/{self.unit[1]}', color=color)
        ax31.plot(self.taxis, self.data[1, :], label=self.label[1], color=color)
        ax31.tick_params(axis='y', labelcolor=color)

        # Making a third axis for proppant concentration
        ax32 = ax3.twinx()
        color = 'red'
        ax32.spines["right"].set_position(("outward", 60))  # Offset the right spine of ax3
        ax32.set_ylabel(f'{self.label[2]}/{self.unit[2]}', color=color)
        ax32.plot(self.taxis, self.data[2,:], label=self.label[2], color=color)
        ax32.tick_params(axis='y', labelcolor=color)
        ax3.xaxis_date()
        # set legend for ax 3
        
        # generate legend
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax31.get_legend_handles_labels()
        lines3, labels3 = ax32.get_legend_handles_labels()

        ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot_all_vars(self, ax3):
        """
        Plot the data of all variables. on ax. 
        """
        
        self.gen_unit()
        # Read the pumping curve data and plot it
        # ax3 = plt.subplot2grid((3,4), (2,0), colspan=4, rowspan=1)
        color = 'blue'
        ax3.set_xlabel('Time')
        ax3.set_ylabel(f'{self.label[0]}/{self.unit[0]}', color=color)
        ax3.plot(self.taxis, self.data[0,:], label=self.label[0], color=color)
        ax3.tick_params(axis='y', labelcolor=color)

        # Making a second axis for treating pressure
        ax31 = ax3.twinx()
        color = 'green'
        ax31.set_ylabel(f'{self.label[1]}/{self.unit[1]}', color=color)
        ax31.plot(self.taxis, self.data[1, :], label=self.label[1], color=color)
        ax31.tick_params(axis='y', labelcolor=color)

        # Making a third axis for proppant concentration
        ax32 = ax3.twinx()
        color = 'red'
        ax32.spines["right"].set_position(("outward", 60))  # Offset the right spine of ax3
        ax32.set_ylabel(f'{self.label[2]}/{self.unit[2]}', color=color)
        ax32.plot(self.taxis, self.data[2,:], label=self.label[2], color=color)
        ax32.tick_params(axis='y', labelcolor=color)
        ax3.xaxis_date()
        # set legend for ax 3
        
        # generate legend
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax31.get_legend_handles_labels()
        lines3, labels3 = ax32.get_legend_handles_labels()

        ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper right')