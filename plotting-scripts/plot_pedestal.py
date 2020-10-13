'''
Plots mean and std of pedestal adc distributions

Usage:
  python3 -i plot_pedestal.py <filename>

'''

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
from scipy.stats import norm,mode

def _key2unique(key, pos=None):
    if isinstance(key, np.ndarray):
        io_group, io_channel, chip_id, channel_id = zip(*[k.split('-') for k in key])
        return unique_channel_id(io_group.astype(int), io_channel.as_type(int), chip_id.as_type(int), channel_id.as_type(int))
    else:
        io_group, io_channel, chip_id, channel_id = key.split('-')        
        return unique_channel_id(int(io_group), int(io_channel), int(chip_id), int(channel_id))

def _unique2key(unique, pos=None):
    channel_id = unique % 64
    chip_id = (unique // 64) % 255
    io_channel = (unique // (64*255)) % 255
    io_group = (unique // (64*255*255)) % 255
    #if isinstance(unique, np.ndarray):
    #    return np.array(['-'.join([str(int(io_group[i])), str(int(io_channel[i])), str(int(chip_id[i])), str(int(channel_id[i]))]) for i in range(len(unique))])
    return '-'.join([str(int(io_group)), str(int(io_channel)), str(int(chip_id)), str(int(channel_id))])

def plot_adc_dist(data, channel, name=None):
    if name is None:
        plt.figure('adc dist channel {} ({})'.format(_unique2key(channel), channel))
    else:
        plt.figure(name)
    plt.hist(data[channel]['adc'], bins=range(0,256), histtype='step')
    plt.xlabel('ADC')
    plt.ylabel('trigger count')

def plot_time_series(data, channel, name=None):
    if name is None:
        plt.figure('adc time series channel {} ({})'.format(_unique2key(channel), channel))
    else:
        plt.figure(name)
    plt.plot(data[channel]['timestamp'], data[channel]['adc'])
    plt.xlabel('timestamp')
    plt.ylabel('ADC')

def fit_adc_dist(data, channel, name=None):
    vals = np.array(data[channel]['adc'])
    mean, sig = norm.fit(vals)
    print('Mean:',mean,'Sigma:',sig)
    x = np.linspace(0,256,1000)
    plot_adc_dist(data, channel, name=name)
    peak = np.sum(vals == mode(vals)[0][0])
    plt.plot(x+0.5, peak * np.exp(-0.5*(x-mean)**2/sig**2), '--', label='fit')
    
def plot_adc_mean(data, bins=None):
    plt.figure('mean adc')
    x = [data[channel]['mean'] for channel in data.keys()]
    if not len(x): return
    if bins is None:
        bins = np.linspace(np.min(x)-1,np.max(x),int(np.max(x)-np.min(x)+1)*3)
    plt.hist(x, bins=bins, histtype='step')
    plt.xlabel('mean ADC')
    plt.ylabel('channel count')

def plot_adc_std(data, bins=None):
    plt.figure('std adc')
    x = [data[channel]['std'] for channel in data.keys()]
    if not len(x): return    
    if bins is None:
        bins = np.linspace(np.min(x)-1,np.max(x),int(np.max(x)-np.min(x)+1)*10)    
    plt.hist([data[channel]['std'] for channel in data.keys()], bins=bins, histtype='step')
    plt.xlabel('std dev ADC')
    plt.ylabel('channel count')

def scatter_adc_std_mean(data):
    plt.figure('scatter adc mean/std')
    plt.scatter([data[channel]['mean'] for channel in data.keys()], [data[channel]['std'] for channel in data.keys()],1)
    plt.xlabel('mean ADC')
    plt.ylabel('std dev ADC')

def plot_summary(data):
    plot_exists = plt.fignum_exists('summary')
    if plot_exists:
        fig = plt.figure('summary')
        axes = fig.axes
    else:
        fig,axes = plt.subplots(2,1,sharex='col',num='summary')
    fig.subplots_adjust(hspace=0)
    
    channels = sorted(data.keys())
    ymean = [data[channel]['mean'] for channel in sorted(data.keys())]
    ystd = [data[channel]['std'] for channel in sorted(data.keys())]

    axes[0].plot(channels,ymean,'.')
    axes[1].plot(channels,ystd,'.')
    axes[1].set(xlabel='unique channel')
    axes[0].set(ylabel='mean ADC')
    axes[1].set(ylabel='std ADC')

    if not plot_exists:
        ax2 = axes[0].secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(_unique2key))
        ax2.set(xlabel='channel key')
    
def unique_channel_id(io_group, io_channel, chip_id, channel_id):
    return channel_id + 64*(chip_id + 255*(io_channel + 255*(io_group)))
    
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
        if len(adc) < 2: continue

        data[channel] = dict(
            channel_mask = channel_mask,
            timestamp = timestamp,
            adc = adc,
            mean = np.mean(adc),
            std = np.std(adc)
            )

        print('chip: {}\tchannel: {}\tmean: {:.02f}\tstd: {:.02f}\tunique: {}'.format((channel//64)%255,channel%64,data[channel]['mean'],data[channel]['std'],channel))

    return data

if __name__ == '__main__':
    data = main(*sys.argv[1:])
    select_channels = list(data.keys())[-4:]
    print(select_channels)
    plot_adc_mean(data)
    plot_adc_std(data)
    for channel in select_channels:
        plot_time_series(data, channel, name='time series')
    plt.legend([_unique2key(channel) for channel in select_channels])
    for channel in select_channels:
        fit_adc_dist(data, channel, name='adc dist')
    plt.legend([_unique2key(channel) for channel in select_channels])        
    plt.ylim((0.1, plt.ylim()[1]*10))
    plt.yscale('log')
    scatter_adc_std_mean(data)
    plot_summary(data)
