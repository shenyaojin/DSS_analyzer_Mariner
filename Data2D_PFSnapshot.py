# Pass the data from PDS to the DSS analyzer.

import numpy as np
from datetime import datetime, timedelta

class Data2D_PFSnapshot:
    def __init__(self):
        self.data = None
        self.taxis = None
        self.daxis = None
        self.start_time = None
        self.history = None

    def load_snapshot