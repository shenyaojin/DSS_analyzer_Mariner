from scipy.signal import butter, lfilter, filtfilt, freqz
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from scipy.stats.stats import pearsonr
import sys


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lppass(freqcut, fs, order=2):
    nyq = 0.5 * fs
    w = freqcut / nyq
    b, a = butter(order, w, btype='low')
    return b, a


def butter_hppass(freqcut, fs, order=2):
    nyq = 0.5 * fs
    w = freqcut / nyq
    b, a = butter(order, w, btype='high')
    return b, a


def bpfilter(data, dt, lowcut, highcut, order=2,axis=-1):
    ''' bpfilter(data, dt, lowcut, highcut, order=2)
	'''
    fs = 1 / dt
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y


def lpfilter(data, dt, freqcut, order=2, plotSpectrum=False,axis=-1):
    fs = 1 / dt
    b, a = butter_lppass(freqcut, fs, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y


def hpfilter(data, dt, freqcut, order=2,axis=-1):
    fs = 1 / dt
    b, a = butter_hppass(freqcut, fs, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y


def amp_spectrum(data, dt, norm=None):
    data = data.flatten()
    N = len(data)
    freqs = np.fft.fftfreq(N, dt)
    asp = np.abs(np.fft.fft(data,norm=norm))
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    asp = asp[idx]
    ind = freqs >= 0
    return freqs[ind], asp[ind]


def samediff(data):
    y = np.diff(data)
    y = np.append(y, y[-1])
    return y


def fillnan(data):
    ind = ~np.isnan(data)
    x = np.array(range(len(data)))
    y = np.interp(x, x[ind], data[ind]);
    return y


def timediff(ts1, ts2):
    tdiff = ts1 - ts2
    if tdiff.days < 0:
        return -(-tdiff).seconds
    else:
        return tdiff.seconds


def diffplot(data1, data2, crange=(-1, 1)):
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(data1, aspect='auto')
    plt.clim(crange)
    plt.subplot(1, 3, 2)
    plt.imshow(data2, aspect='auto')
    plt.clim(crange)
    plt.subplot(1, 3, 3)
    plt.imshow(data1 - data2, aspect='auto')
    plt.clim(crange)
    plt.show()


def get_interp_mat(anchor_N, N, kind='quadratic'):
    x = np.arange(N)
    anchor_x = np.linspace(x[0], x[-1], anchor_N)
    interp_mat = np.zeros((N, anchor_N))
    for i in range(anchor_N):
        test_y = np.zeros(anchor_N, )
        test_y[i] = 1
        col = interp1d(anchor_x, test_y, kind=kind)(x)
        interp_mat[:, i] = col
    return interp_mat


def get_interp_mat_anchorx(x, anchor_x, kind='quadratic'):
    anchor_N = len(anchor_x)
    N = len(x)
    interp_mat = np.zeros((N, anchor_N))
    for i in range(anchor_N):
        test_y = np.zeros(anchor_N, )
        test_y[i] = 1
        col = interp1d(anchor_x, test_y, kind=kind)(x)
        interp_mat[:, i] = col
    return interp_mat


def get_smooth_curve(x0, anchor_x, data, kind='quadratic', errstd=3, iterN=2):
    iterdata = data.copy()
    iterx = x0.copy()
    for i in range(iterN + 1):
        interp_mat = get_interp_mat_anchorx(iterx, anchor_x, kind=kind)
        x = np.linalg.lstsq(interp_mat, iterdata)[0]
        err = np.abs(iterdata - np.dot(interp_mat, x)) ** 2
        goodind = err < np.std(err) * errstd
        iterdata = iterdata[goodind]
        iterx = iterx[goodind]
    interp_mat = get_interp_mat_anchorx(x0, anchor_x, kind=kind)
    smdata = np.dot(interp_mat, x)
    return smdata, x


def rms(a,axis=None):
    return np.sqrt(np.mean((a) ** 2,axis=axis))

def matdatenum_to_pydatetime(matlab_datenum):
    python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(
        days=366)
    return python_datetime


def rfft_xcorr(x, y):
    M = len(x) + len(y) - 1
    N = 2 ** int(np.ceil(np.log2(M)))
    X = np.fft.rfft(x, N)
    Y = np.fft.rfft(y, N)
    cxy = np.fft.irfft(X * np.conj(Y))
    cxy = np.hstack((cxy[:len(x)], cxy[N - len(y) + 1:]))
    return cxy


def xcor_match(a, b):
    x = a.copy()
    ref = b.copy()
    x -= np.mean(x)
    ref -= np.mean(ref)
    r = pearsonr(x, ref)[0]
    if abs(r) < 0.3:
        return np.nan
    cxy = rfft_xcorr(x, ref)
    index = np.argmax(cxy)
    if index < len(x):
        return index
    else:  # negative lag
        return index - len(cxy)


def timeshift_xcor(data1, data2, winsize, step=1, lowf=1 / 10):
    N = len(data1)
    ori_x = np.arange(N)
    cx = np.arange(np.int(winsize / 2), np.int(N - winsize / 2 - 1), step)
    cy = np.zeros(len(cx))
    for i in range(len(cx)):
        winbg = cx[i] - np.int(winsize / 2)
        wined = cx[i] + np.int(winsize / 2)
        cy[i] = xcor_match(data1[winbg:wined], data2[winbg:wined])
    ind = ~np.isnan(cy)
    ts = np.interp(ori_x, cx[ind], cy[ind])
    ts = lpfilter(ts, 1, lowf)
    tar_x = ori_x + ts
    shift_data1 = np.interp(tar_x, ori_x, data1)
    return ts, shift_data1


def running_average(data, N):
    outdata = np.convolve(data, np.ones((N,)) / N, mode='same')
    halfN = int(N / 2)
    for i in range(halfN + 1):
        outdata[i] = np.mean(data[:i + halfN])
    for i in range(1, halfN + 1):
        outdata[-i] = np.mean(data[-i - halfN:])
    return outdata


def degC_to_degK(C):
    return C - 273.15


def degC_to_degF(C):
    return C * 9 / 5 + 32


def degF_to_degC(C):
    return (C - 32) * 5 / 9


def degK_to_degF(K):
    return K * 9 / 5 - 459.67


def degF_to_degK(F):
    return (F + 459.67) * 5 / 9


def print_progress(n):
    sys.stdout.write("\r" + str(n))
    sys.stdout.flush()


def phase_wrap(data):
    phase = np.angle(np.exp(1j * data))
    return phase

def dist_3D(x, y, z, x1, y1, z1):
    r = ((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) ** 0.5
    return r


def dummy_fun():
    print('hello world')


def robust_polyfit(x,y,order=1,errtol=2):
    para = np.polyfit(x,y,order)
    errs = np.abs(y-np.polyval(para,x))
    err_threshold = np.std(errs)*errtol
    goodind = errs<err_threshold
    para = np.polyfit(x[goodind],y[goodind],order)
    return para

from dateutil.parser import parse
from datetime import timedelta
def fetch_timestamp_fast(timestamp_strs, downsampling=100):
    x = np.arange(len(timestamp_strs))
    x_sparse = list(map(int,np.round(np.linspace(0,x[-1],len(x)//downsampling))))
    ts_sparse = np.array(list(map(parse,timestamp_strs[x_sparse])))
    t_sparse = np.array([(t-ts_sparse[0]).total_seconds() for t in ts_sparse])
    t = np.interp(x,x_sparse,t_sparse)
    ts = [(ts_sparse[0]+timedelta(seconds=dt)) for dt in t]
    return ts,t

def multi_legend(lns,loc='best'):
    labs = [l.get_label() for l in lns]
    plt.legend(lns,labs,loc=loc)


def datetime_interp(timex,timex0,y0):
    x = [(t-timex0[0]).total_seconds() for t in timex]
    x0 = [(t-timex0[0]).total_seconds() for t in timex0]
    return np.interp(x,x0,y0)
    
def running_average(data,N):
    return np.convolve(data,np.ones((N,))/N,mode='same')