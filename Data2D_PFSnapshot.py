# Pass the data from PDS to the DSS analyzer.
# Analyzer of the synthetic data from the PDS. Since it has no time stamp info
# So I need a new class to handle the data. And that's why I write this class.

import numpy as np
from datetime import datetime, timedelta

class Data2D_PFSnapshot:
    def __init__(self):
        self.data = None
        self.taxis = None
        self.daxis = None
        self.start_time = None
        self.history = None

    import numpy as np

    def load_snapshot(self, **kwargs):
        """
        Load a snapshot from either direct data parameters or from a file.
        Expects:
          - data, taxis, daxis in kwargs, OR
          - file in kwargs pointing to a .npz file containing 'data', 'taxis', and 'daxis'.

        If some parameters are missing, logs a warning and sets them to None.
        """

        # Case 1: Data is passed directly
        if 'data' in kwargs:
            self.data = kwargs['data']

            if 'taxis' in kwargs:
                self.taxis = kwargs['taxis']
            else:
                print('Warning: "taxis" not provided, setting to None.')
                self.taxis = None

            if 'daxis' in kwargs:
                self.daxis = kwargs['daxis']
            else:
                print('Warning: "daxis" not provided, setting to None.')
                self.daxis = None

        # Case 2: A file is provided
        elif 'file' in kwargs:
            try:
                tmp_datafile = np.load(kwargs['file'])
            except IOError:
                print(f'Error: Could not open file {kwargs["file"]}.')
                return

            if 'data' in tmp_datafile:
                self.data = tmp_datafile['data']
            else:
                print('Warning: "data" not found in file, setting to None.')
                self.data = None

            if 'taxis' in tmp_datafile:
                self.taxis = tmp_datafile['taxis']
            else:
                print('Warning: "taxis" not found in file, setting to None.')
                self.taxis = None

            if 'daxis' in tmp_datafile:
                self.daxis = tmp_datafile['daxis']
            else:
                print('Warning: "daxis" not found in file, setting to None.')
                self.daxis = None

        # Case 3: Neither data nor file was provided
        else:
            print('No data is provided. Please provide either "data" or "file".')
            return

    def get_real_time_info(self, **kwargs):

        return 0