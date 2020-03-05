'''
Plots trigger rate per channel

Usage:
  python3 -i plot_rate.py <filename>

'''

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_trigger_rate(data, channel, name=None):
    if name is None:
        plt.figure('adc dist channel {}'.format(channel))
    else:
        plt.figure(name)
    t_min = int(min(data[channel]['timestamp_s']))
    t_max = int(max(data[channel]['timestamp_s']))
    plt.hist(data[channel]['timestamp_s']-t_min, weights=[1/5]*len(data[channel]['timestamp_s']), bins=np.linspace(0, t_max-t_min, int((t_max-t_min)/5)), histtype='step')
    plt.xlabel('time [s]')
    plt.ylabel('trigger rate [Hz]')
    plt.legend(['channel {}'.format(channel)])

def plot_trigger_rate_summary(data):
    plt.figure('trigger rate summary')
    channels = list(set(data.keys()))
    plt.plot(channels, [data[channel]['rate'] for channel in data], '.')
    plt.xlabel('channel')
    plt.ylabel('trigger rate [Hz]')

def main(*args):
    filename = args[0]

    print('opening',filename)

    plt.ion()

    f = h5py.File(filename,'r')

    channels = set(f['packets'][:]['channel_id'])
    valid_parity_mask = (f['packets'][:]['valid_parity'] == 1)

    data = dict()
    for channel in channels:
        channel_mask = f['packets'][:]['channel_id'] == channel
        timestamp = f['packets'][np.logical_and(channel_mask, valid_parity_mask)]['timestamp']
        timestamp_s = timestamp * 100e-9

        data[channel] = dict(
            channel_mask = channel_mask,
            timestamp = timestamp,
            timestamp_s = timestamp_s,
            rate = len(timestamp) / (max(timestamp_s) - min(timestamp_s))
            )

    return data

if __name__ == '__main__':
    data = main(*sys.argv[1:])
    plot_trigger_rate(data, 0)
    plot_trigger_rate_summary(data)

