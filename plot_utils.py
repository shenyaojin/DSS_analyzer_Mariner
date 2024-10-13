import numpy as np
import matplotlib.pyplot as plt
# dependence: DSS_analyzer_Mariner; JIN_pylib

class fracture_investigation:
    # DSS data, must be the Data2D_DSS class for Mariner's DSS data has a different format. 
    def __init__(self, fig, data, extent, datetime_data_crop, gauge_data_downsample_crop, 
    cum_strain_stimulation, DASdata, cum_strain_production, fracture_hit_depth_data): 
        self.data = data
        self.datetime_data_crop = datetime_data_crop
        self.gauge_data_downsample_crop = gauge_data_downsample_crop
        self.cum_strain_stimulation = cum_strain_stimulation
        self.DASdata = DASdata
        self.cum_strain_production = cum_strain_production
        self.fracture_hit_depth_data = fracture_hit_depth_data

        self.pick_t = np.median(data.taxis)
        self.pick_d = np.median(data.daxis)
        self.trc_lim = np.array([-0.5,4])
        self.fig = fig
        self.ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
        self.ax2 = plt.subplot2grid((6, 6), (4, 0), rowspan=2, colspan=4, sharex=self.ax1)
        self.ax3 = plt.subplot2grid((6, 6), (0, 4), rowspan=4, colspan=2, sharey=self.ax1)
        self.hline = None
        self.ax3hline = None
        self.vline = None
        self.extent = extent
        self.ori_xlim = None
        self.ori_ylim = None
        self.xlim = None
        self.ylim = None
        self.pending_zoom_x = False
        self.pending_zoom_y = False
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def draw(self): # 
        
        for fracloc in self.fracture_hit_depth_data:
            self.ax1.axhline(fracloc, color='black', linestyle='--')
        self.ax1.imshow(self.data.data, cmap = 'jet', aspect = 'auto', extent = self.extent)
        self.ax1.set_yticks([])
        self.ax1.set_ylabel("MD/ft")
        self.update_trace_plots()
        if self.xlim is None:
            self.xlim = self.ax1.axis()[:2]
            self.ylim = self.ax1.axis()[2:]
            self.ori_xlim = self.xlim
            self.ori_ylim = self.ylim

    def detect_trc_ylim(self, trc): 
        max_val = np.max(trc)
        min_val = np.min(trc)
        self.trc_lim = np.array([min_val, max_val])

    def update_trace_plots(self):
        # remove the previous hline
        if self.hline:
            self.hline.remove()
        if self.ax3hline:
            self.ax3hline.remove()

        md, trc = self.data.get_value_by_depth(self.pick_d)

        self.hline = self.ax1.axhline(self.pick_d, color='red', linestyle='--')

        self.ax2.clear()
        self.ax2.plot(self.data.datetimestamp, trc, color='red', label='DSS data')
        self.detect_trc_ylim(trc)
        ax21 = self.ax2.twinx()
        ax21.plot(self.datetime_data_crop, self.gauge_data_downsample_crop, color='green', label='Pressure curve')
        self.ax2.set_ylim(self.trc_lim)
        self.ax2.set_xlabel("DATE")

        self.ax3.clear()
        self.ax3.plot(self.cum_strain_stimulation, self.DASdata.mds, color='cyan')
        self.ax3.set_ylim([self.DASdata.daxis[-1], self.DASdata.daxis[0]])
        #self.ax3.set_yticks([])
        ax31 = self.ax3.twiny()
        ax31.plot(self.cum_strain_production, self.data.mds, color='magenta')
        for fracloc in self.fracture_hit_depth_data:
            self.ax3.axhline(fracloc, color='black', linestyle='--')
        self.ax3hline = self.ax3.axhline(self.pick_d, color='red', linestyle='--')
        self.fig.canvas.draw()
    
    def update_zoom(self):
        self.ax1.set_xlim(self.xlim)
        self.ax1.set_ylim(self.ylim)
        self.fig.canvas.draw()
    
    def _clear_pending(self):
        self.pending_zoom_x = False
        self.pending_zoom_y = False
    
    def on_key_press(self, event):
        if event.key == 'a':
            self.pick_d = event.ydata
            self.update_trace_plots()

        if event.key == '=':
            self.trc_lim = self.trc_lim*1.2
            self.update_trace_plots()

        if event.key == '-':
            self.trc_lim = self.trc_lim/1.2
            self.update_trace_plots()

        if event.key == 'o':
            self.xlim = self.ori_xlim
            self.ylim = self.ori_ylim
            self.update_zoom()
        
        if event.key == 'j':
            # move to larger md
            self.pick_d = self.pick_d + 1
            self.update_trace_plots()

        if event.key == 'k':
            # move to smaller md
            self.pick_d = self.pick_d - 1
            self.update_trace_plots()
        
        self._clear_pending()