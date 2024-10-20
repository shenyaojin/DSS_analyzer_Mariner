from dataclasses import dataclass
from . import gjsignal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from scipy.signal import medfilt2d,tukey
import matplotlib.dates as mdates
from dateutil.parser import parse
from copy import copy
import h5py
import copy
import pickle
from matplotlib.ticker import MaxNLocator

# Basically is a modified version based on my own habit. ???????
class Data2D():

    def __init__(self):
        self.data = None   # data, 2D array
        self.start_time = None  # starting time using datetime
        self.taxis = []  # time axis in second from start_time
        self.chans = [] # fiber channel number
        self.daxis = []  # fiber physical distance or location
        self.mds = []  # fiber physical distance or location. I don't know why I have two of them. :)

    def set_data(self,data):
        self.data = data

    def rotate_data(self): 
        self.data = self.data.T
    
    def apply_timeshift(self,ts):
        self.start_time += timedelta(hours=ts)
    
    # Time stamp stuff
    def set_datetimestamp(self, timestamp): 
        # time stamp must be a datetime array
        timestamp = np.array(timestamp)
        self.datetimestamp = timestamp

    def cal_timestamp_from_taxis(self):
        timestamps = [self.start_time + timedelta(seconds=t) 
            for t in self.taxis]
        self.timestamps = timestamps
    
    def print_timestamp(self):
        self.cal_timestamp_from_taxis()
        return self.timestamps

    def set_mds(self,mds):
        self.mds = mds
    
    def _check_inputtime(self,t,t0):
        out_t = t
        if t is None:
            out_t = t0
        if type(t) is datetime:
            out_t = (t-self.start_time).total_seconds()
        return out_t
    
    def reset_starttime(self):
        self.start_time += timedelta(seconds=self.taxis[0])
        self.taxis -= self.taxis[0]

    def select_time(self,bgtime,edtime,makecopy=False,reset_starttime=True): # modified for Marina project
        bgt = self._check_inputtime(bgtime,self.taxis[0])
        edt = self._check_inputtime(edtime,self.taxis[-1])
        
        ind = (self.taxis>=bgt)&(self.taxis<=edt)
        if makecopy:
            out_data = copy.copy(self)
            out_data.taxis = self.taxis[ind]
            out_data.datetimestamp = self.datetimestamp[ind]
            if reset_starttime: 
                out_data.start_time += timedelta(seconds=out_data.taxis[0])
                out_data.taxis -= out_data.taxis[0]
            out_data.data = out_data.data[:, ind]
            return out_data
        else:
            self.taxis = self.taxis[ind]
            self.datetimestamp = self.datetimestamp[ind]
            if reset_starttime:
                self.start_time += timedelta(seconds=self.taxis[0])
                self.taxis -= self.taxis[0]
            self.data = self.data[:, ind]

    def select_depth(self,bgdp,eddp,makecopy=False,ischan=False):
        
        if ischan:
            dists = self.chans
        else:
            dists = self.daxis
        bgt = self._check_inputtime(bgdp,dists[0])
        edt = self._check_inputtime(eddp,dists[-1])
        
        ind = (dists>=bgdp)&(dists<=eddp)
        if makecopy:
            out_data = copy.copy(self)
            out_data.data = out_data.data[ind,:]
            try:
                out_data.daxis =out_data.daxis[ind]
                out_data.mds =out_data.mds[ind]
            except: 
                pass
            try:
                out_data.chans =out_data.chans[ind]
                out_data.mds =out_data.mds[ind]
            except: 
                pass

            return out_data
        else:
            self.data = self.data[ind,:]
            try:
                self.daxis =self.daxis[ind]
            except: 
                pass
            try:
                self.chans =self.chans[ind]
            except: 
                pass
    
    def copy(self):
        return copy.deepcopy(self)
    
    def set_chans(self,chans):
        self.chans = chans
    
    def median_filter(self,kernel_size=(5,3)):
        self.data = medfilt2d(self.data,kernel_size=kernel_size)
        self.history.append('median_filter(kernel_size={})'.format(str(kernel_size)))
    
    def window_data_time(self,bgtime, edtime,reset_startime=True):
        ind = (self.taxis>bgtime)&(self.taxis<edtime)
        self.data = self.data[:,ind]
        self.taxis = self.taxis[ind]
        t0 = self.taxis[0]
        if reset_startime:
            self.taxis = self.taxis-t0
            self.start_time += timedelta(seconds=t0)
    
    def window_data_depth(self,bgmd,edmd,ismd=True):
        if ismd:
            ind = (self.mds>bgmd)&(self.mds<edmd)
        else:
            ind = (self.chans>bgmd)&(self.chans<edmd)
        self.data = self.data[ind,:]
        try:
            self.mds = self.mds[ind]
        except:
            print('cannot find mds field')
            pass
        try:
            self.chans = self.chans[ind]
        except:
            print('cannot find chans field')
            pass
    
    def lp_filter(self,corner_freq,order=2,axis=1):
        if axis == 1:
            dt = np.median(np.diff(self.taxis))
        if axis == 0:
            dt = np.median(np.diff(self.mds))
        self.data = gjsignal.lpfilter(self.data,dt,corner_freq,order=order,axis=axis)
        self.history.append('lp_filter(corner_freq={},order={},axis={})'
                .format(corner_freq,order,axis))

    def hp_filter(self,corner_freq,order=2,axis=1,edge_taper=0.1):
        if axis == 1:
            dt = np.median(np.diff(self.taxis))
        if axis == 0:
            dt = np.median(np.diff(self.mds))
        self.data *= tukey(self.data.shape[1],edge_taper).reshape((1,-1))
        self.data = gjsignal.hpfilter(self.data,dt,corner_freq,order=order,axis=axis)
        self.history.append('hp_filter(corner_freq={},order={},axis={})'
                .format(corner_freq,order,axis))

    def bp_filter(self,lowf,highf,order=2,axis=1,edge_taper=0.1):
        if axis == 1:
            dt = np.median(np.diff(self.taxis))
        if axis == 0:
            dt = np.median(np.diff(self.mds))
        self.data *= tukey(self.data.shape[1],edge_taper).reshape((1,-1))
        self.data = gjsignal.bpfilter(self.data,dt,lowf,highf,order=order,axis=axis)
        self.history.append('bp_filter(lowf={},highf={},order={},axis={})'
                .format(lowf,highf,order,axis))
    
    def down_sample(self,ds_R):
        dt = np.median(np.diff(self.taxis))
        self.lp_filter(1/dt/2/ds_R*0.8)
        self.data = self.data[:,::ds_R]
        self.taxis = self.taxis[::ds_R]
        self.history.append('down_sample({})'.format(ds_R))

    def take_time_diff(self):
        data = np.diff(self.data,axis=1)
        data = data/np.diff(self.taxis).reshape((1,-1))
        data = np.hstack((np.zeros((data.shape[0],1)),data))
        self.data = data
        self.history.append('take_diff()')
    
    def apply_gauge_length(self,gauge_chan_num=1):
        strain_data = self.data[gauge_chan_num:,:]-self.data[:-gauge_chan_num,:]
        strain_data /= (self.daxis[gauge_chan_num:]-self.daxis[:-gauge_chan_num]).reshape((-1,1))
        self.data = strain_data
        self.daxis = (self.daxis[gauge_chan_num:]+self.daxis[:-gauge_chan_num])/2
        self.history.append(f'apply_gauge_length(gauge_chan_num={gauge_chan_num})')
    
    def cumsum(self,axis=1): # to get the strain change of the data.
        data = np.cumsum(self.data,axis=axis)
        if axis==1:
            ds = np.diff(self.taxis)
            ds = np.concatenate(([1],ds))
            data = data*ds.reshape((1,-1))
        if axis==0:
            ds = np.diff(self.mds)
            ds = np.concatenate(([1],ds))
            data = data*ds.reshape((-1,1))

        self.data = data
        self.history.append(f'cumsum(axis={axis})')
    
    def plot_simple_waterfall(self,downsample = [1,1]):
        extent = [0,self.data.shape[1],self.data.shape[0],0]
        plt.imshow(self.data[::downsample[0],::downsample[1]]
                ,cmap=plt.get_cmap('bwr'),aspect='auto',extent=extent)
    
    def get_extent(self,ischan=False,timescale='second',use_timestamp=False):
        xlim = np.array([self.taxis[0],self.taxis[-1]])
        if timescale == 'hour':
            xlim = xlim/3600
        if timescale == 'day':
            xlim = xlim/3600/24
        if ischan:
            ylim = [self.chans[-1],self.chans[0]]
        else:
            ylim = [self.mds[-1],self.mds[0]]
        if use_timestamp:
            edtime = self.start_time + timedelta(seconds=self.taxis[-1])
            bgtime = self.start_time + timedelta(seconds=self.taxis[0])
            xlim = [bgtime,edtime]
            xlim = mdates.date2num(xlim)
        extent = [xlim[0],xlim[-1],ylim[0],ylim[-1]]
        return extent

    def plot_waterfall(self,ischan = False, cmap=plt.get_cmap('bwr') 
            , timescale='second',use_timestamp=False
            ,downsample=[1,1]
            ,xaxis_rotation=0
            ,xtickN = 4
            ,timefmt = '%m/%d\n%H:%M:%S.{ms}' 
            ,timefmt_ms_precision = 1
            ):
        '''
        timescale options: 'second','hour','day'
        '''
        extent = self.get_extent(ischan=ischan
            ,timescale=timescale,use_timestamp=use_timestamp)
        plt.imshow(self.data[::downsample[0],::downsample[1]]
                ,cmap = cmap, aspect='auto',extent=extent)
        if use_timestamp:
            plt.gca().xaxis_date()
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(xtickN))
            plt.xticks(rotation=xaxis_rotation)

    # add a function to enable co plot with other data. return the handle of the plot
    def plot_water_on_ax(self, ax, ischan=False, cmap=plt.get_cmap('bwr'),
                     timescale='second', use_timestamp=False, downsample=[1, 1],
                     xaxis_rotation=0, xtickN=4, timefmt='%m/%d\n%H:%M:%S.{ms}',
                     timefmt_ms_precision=1):
        '''
        timescale options: 'second', 'hour', 'day'
        '''
        extent = self.get_extent(ischan=ischan, timescale=timescale, use_timestamp=use_timestamp)
        img = ax.imshow(self.data[::downsample[0], ::downsample[1]],
                cmap=cmap, aspect='auto', extent=extent)
        if use_timestamp:
            ax.xaxis_date()
            ax.xaxis.set_major_locator(MaxNLocator(xtickN))
            ax.tick_params(axis='x', labelrotation=xaxis_rotation)
        return img        
    
    def plot_wiggle(self,scale=1,trace_step = 1,linewidth=1):
        # Extract the data, time axis, and distance axis from the seismic_data object
        data = self.data
        taxis = self.taxis
        daxis = self.daxis

        # Get the number of time and distance points
        nt = len(taxis)
        nd = len(daxis)

        # Loop over each trace
        for i in range(0,nd,trace_step):
            # Scale and shift the data for this trace
            trace = data[i, :] * scale + daxis[i]

            # Plot the trace as a line
            plt.plot(trace, taxis, color='k',linewidth=linewidth)

            # Fill between the trace and the zero line
            plt.fill_betweenx(taxis, daxis[i], trace, where=(trace>daxis[i]), color='r', linewidth=0.5*linewidth)
            plt.fill_betweenx(taxis, daxis[i], trace, where=(trace<daxis[i]), color='b', linewidth=0.5*linewidth)
        
        plt.gca().invert_yaxis()

    
    def fill_gap_zeros(self,fill_value=0,dt=None):
        """
        Filling data gap with zeros or with a fixed value
        """
        if dt is None:
            dt = np.median(np.diff(self.taxis))
        N = int(np.round((np.max(self.taxis)-np.min(self.taxis))/dt))+1
        new_taxis = np.linspace(np.min(self.taxis),np.max(self.taxis)+dt,N)
        new_data = np.zeros((self.data.shape[0],N))
        new_data[:,:] = fill_value
        for i in range(self.data.shape[1]):
            ind = int(np.round(self.taxis[i]/dt))
            new_data[:,ind] = self.data[:,i]
        self.data = new_data
        self.taxis = new_taxis
        self.history.append(f'fill_gap_zeros(fill_value={fill_value},dt={dt})')

    def fill_gap_interp(self,dt=None):
        if dt is None:
            dt = np.median(np.diff(self.taxis))
        N = int(np.round((np.max(self.taxis)-np.min(self.taxis))/dt))+1
        new_taxis = np.linspace(np.min(self.taxis),np.max(self.taxis),N)
        new_data = np.zeros((self.data.shape[0],N))
        for i in range(self.data.shape[0]):
            new_data[i,:] = np.interp(new_taxis,self.taxis,self.data[i,:],left=0,right=0)
        self.data = new_data
        self.taxis = new_taxis
        self.history.append(f'fill_gap_interp(dt={dt})')

    def interp_time(self,new_taxis):
        new_data = np.zeros((self.data.shape[0],len(new_taxis)))
        for i in range(self.data.shape[0]):
            new_data[i,:] = np.interp(new_taxis,self.taxis,self.data[i,:],left=0,right=0)
        self.data = new_data
        self.taxis = new_taxis
    
    def get_value_by_depth(self,depth):
        ind = np.argmin(np.abs(self.mds-depth))
        md = self.mds[ind]
        return md,self.data[ind,:]

    def get_value_by_timestr(self,timestr,fmt=None):
        if fmt is None:
            t = parse(timestr)
        else:
            t = datetime.strptime(timestr,fmt)
        dt = (t-self.start_time).total_seconds()
        ind = np.argmin(np.abs(self.taxis-dt))
        output_time = self.start_time + timedelta(seconds=self.taxis[ind])
        return output_time,self.data[:,ind]
    
    def get_value_by_time(self,t):
        ind = np.argmin(np.abs(self.taxis-t))
        actual_t = self.taxis[ind]
        return actual_t,self.data[:,ind]
    
    def make_audio_file(self,filename,bgdp=None,eddp=None):
        from scipy.io.wavfile import write
        DASdata = self.select_depth(bgdp,eddp,makecopy=True)
        rate = int(1/np.median(np.diff(DASdata.taxis)))
        data = np.mean(DASdata.data,axis=0)
        scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        write(filename, rate, scaled)
        return scaled
    
    # I/O functions: with npz and other data2D objects

    # ONLY USE THIS FOR PACKING DATA!
    def savenpz(self, filename):
        serialized_file = pickle.dumps(self)
        np_serialized_a = np.array([serialized_file], dtype=np.void)
        np.savez(filename, data=np_serialized_a)

    def loadnpz(self, filename):
        loaded_data = np.load(filename)
        serialized_a = loaded_data['data'][0]
        new_instance = pickle.loads(serialized_a.tobytes())
        self.__dict__.update(new_instance.__dict__)

    def right_merge(self,data):
        taxis = data.taxis + (data.start_time - self.start_time).total_seconds()
        self.taxis = np.concatenate((self.taxis,taxis))
        self.data = np.concatenate((self.data.T,data.data.T)).T
    
    def quick_populate(self,data,dt,dx):
        self.data = data
        self.taxis = np.arange(data.shape[1])*dt
        self.daxis = np.arange(data.shape[0])*dx

def merge_data2D(data_list):
    data_list = np.array(data_list)
    bgtime_lst = np.array([d.start_time for d in data_list])
    ind = np.argsort(bgtime_lst)
    bgtime_lst = bgtime_lst[ind]
    data_list = data_list[ind]

    t_samples = [d.data.shape[1] for d in data_list]
    N_samples = np.sum(t_samples)

    bgtime = data_list[0].start_time
    taxis_list = [d.taxis + (d.start_time-bgtime).total_seconds() for d in data_list]

    merge_data = copy.deepcopy(data_list[0])
    merge_data.data = np.concatenate([d.data.T for d in data_list]).T
    merge_data.taxis = np.concatenate(taxis_list)
    return merge_data