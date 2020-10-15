import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import norm, mode
import sys

threshold = 128
gain = 4 # mV /ke-
runtime = 120
lsb = 3.91
#vref = 1.546 V
#vcm = 544 mV

nonrouted_channels = [6,7,8,9,
                      22,23,24,25,
                      38,39,40,
                      54,55,56,57]

def unique_channel_id(io_group, io_channel, chip_id, channel_id):
    return channel_id + 64*(chip_id + 255*(io_channel + 255*(io_group)))

def plot_summary(data):
    parameters = {'axes.labelseize' : 15,
                  'xtick.labelsize' : 15,
                  'ytick.labelsize' : 15 }
    plot_exists = plt.fignum_exists('summary')
    if plot_exists:
        fig = plt.figure('summary')
        axes = fig.axes
    else:
        fig,axes = plt.subplots(2,1,sharex='col',num='summary')
    fig.subplots_adjust(hspace=0)

    channels = [ (channel//64)%255 for channel in sorted(data.keys()) if channel%64 not in nonrouted_channels]
    ch_rate = [data[channel]['rate'] for channel in sorted(data.keys()) if channel%64 not in nonrouted_channels]
    ch_leakage = [data[channel]['leakage'] for channel in sorted(data.keys()) if channel%64 not in nonrouted_channels]

    axes[0].plot(channels,ch_rate,'.')
    axes[1].plot(channels,ch_leakage,'.')
    axes[1].set(xlabel='Chip ID')
    axes[0].set(ylabel='Rate [Hz]')
    axes[1].set(ylabel='Leakage Current [e- / ms] ')
    axes[0].set(yscale='log')
    axes[1].set(yscale='log')

    

def main(*args):
    filename = args[0]
    print('opening',filename)
    plt.ion()
    f = h5py.File(filename,'r')

    data_mask = f['packets'][:]['packet_type'] == 0
    valid_parity_mask = f['packets'][data_mask]['valid_parity'] == 1
    good_data = (f['packets'][data_mask])[valid_parity_mask]

    io_group = good_data['io_group'].astype(np.uint64)
    io_channel = good_data['io_channel'].astype(np.uint64)
    chip_id = good_data['chip_id'].astype(np.uint64)
    channel_id = good_data['channel_id'].astype(np.uint64)
    unique_channels = set(unique_channel_id(io_group, io_channel, chip_id, channel_id))
    
    data = dict()
    for channel in sorted(unique_channels):
        channel_mask = unique_channel_id(io_group, io_channel, chip_id, channel_id) == channel
        timestamp = good_data[channel_mask]['timestamp']
        adc = good_data[channel_mask]['dataword']
        rate_i = len(adc) / runtime

        data[channel] = dict(
            channel_mask = channel_mask,
            timestamp = timestamp,
            adc = adc,
            rate = rate_i,
            leakage = (rate_i)*threshold*lsb*(1000/gain)/1000 # e- / ms
            )
        if rate_i > 20:
            print('chip: {}\tchannel: {}\trate: {:.02f}\tleakage: {:.02f}'.format((channel//64)%255, channel%64, data[channel]['rate'], data[channel]['leakage']))
    return data
                  
if __name__ == '__main__':
    data = main(*sys.argv[1:])
    plot_summary(data)
