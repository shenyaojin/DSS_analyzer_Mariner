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

    def load_snapshot(self, **kwargs):
        # Load the file from data, or using the data directly.
        if 'data' in kwargs:
            self.data = kwargs['data']
            # Also, set the time axis and the distance axis.
            self.taxis = kwargs['taxis']
            self.daxis = kwargs['daxis']
        elif 'file' in kwargs:
            self.data = np.load(kwargs['file'])
        else:
            print('No data is provided.')
            return