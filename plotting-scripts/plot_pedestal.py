'''
Plots mean and std of pedestal adc distributions

Usage:
  python3 -i plot_pedestal.py <filename>

'''

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_adc_dist(data, channel, name=None):
    if name is None:
        plt.figure('adc dist channel {}'.format(channel))
    plt.hist(data[channel]['adc'], bins=range(0,256), histtype='step')
    plt.xlabel('ADC')
    plt.ylabel('trigger count')
    plt.legend(['channel {}'.format(channel)])

def plot_adc_mean(data, bins=50):
    plt.figure('mean adc')
    plt.hist([data[channel]['mean'] for channel in data.keys()], bins=bins, histtype='step')
    plt.xlabel('mean ADC')
    plt.ylabel('channel count')

def plot_adc_std(data, bins=50):
    plt.figure('std adc')
    plt.hist([data[channel]['std'] for channel in data.keys()], bins=bins, histtype='step')
    plt.xlabel('std dev ADC')
    plt.ylabel('channel count')

def scatter_adc_std_mean(data):
    plt.figure('scatter adc mean/std')
    plt.scatter([data[channel]['mean'] for channel in data.keys()], [data[channel]['std'] for channel in data.keys()],1)
    plt.xlabel('mean ADC')
    plt.ylabel('std dev ADC')

def main(*args):
    filename = args[0]

    print('opening',filename)

    plt.ion()

    f = h5py.File(filename,'r')

    channels = set(f['packets'][:]['channel_id'])
    valid_parity_mask = f['packets'][:]['valid_parity'] == 1

    data = dict()
    for channel in channels:
        channel_mask = f['packets'][valid_parity_mask]['channel_id'] == channel
        timestamp = f['packets'][channel_mask]['timestamp']
        adc = f['packets'][channel_mask]['dataword']

        data[channel] = dict(
            channel_mask = channel_mask,
            timestamp = timestamp,
            adc = adc,
            mean = np.mean(adc),
            std = np.std(adc)
            )

        print('channel: {}\tmean: {:.02f}\tstd: {:.02f}'.format(channel,data[channel]['mean'],data[channel]['std']))

    return data

if __name__ == '__main__':
    data = main(*sys.argv[1:])
    plot_adc_dist(data, 0)
    plot_adc_mean(data, bins=np.linspace(0,50,50))
    plot_adc_std(data, bins=np.linspace(0,10,50))
    scatter_adc_std_mean(data)
