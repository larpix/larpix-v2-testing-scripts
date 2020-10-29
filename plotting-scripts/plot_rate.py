'''
Plots trigger rate per channel

Usage:
  python3 -i plot_rate.py <filename>

'''

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import plot_pedestal as larpix_plot
import matplotlib.ticker as ticker

def plot_trigger_rate(data, channel, name=None):
    if name is None:
        plt.figure('trigger rate channel {}'.format(channel))
    else:
        plt.figure(name)
    t_min = int(min(data[channel]['timestamp_s']))
    t_max = int(max(data[channel]['timestamp_s']))
    plt.hist(data[channel]['timestamp_s']-t_min, weights=[1/5]*len(data[channel]['timestamp_s']), bins=np.linspace(0, t_max-t_min, int((t_max-t_min)/5)), histtype='step')
    plt.xlabel('time [s]')
    plt.ylabel('trigger rate [Hz]')
    plt.legend(['channel {}'.format(channel)])

def plot_trigger_rate_summary(data):
    plot_exists = plt.fignum_exists('trigger rate summary')
    fig = plt.figure('trigger rate summary')

    channels = list(set(data.keys()))
    plt.plot(channels, [data[channel]['rate'] for channel in data], '.')
    plt.xlabel('channel')
    plt.ylabel('trigger rate [Hz]')

    if not plot_exists:
        ax2 = fig.axes[0].secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(larpix_plot._unique2key))
        ax2.set(xlabel='channel key')

    plt.yscale('log')
    plt.tight_layout()

def main(*args):
    filename = args[0]

    print('opening',filename)

    plt.ion()

    f = h5py.File(filename,'r')

    if np.sum(f['packets'][:]['local_fifo']) != 0 or np.sum(f['packets'][:]['shared_fifo']) != 0:
        print('FIFO full flag(s)! local: {}\tshared: {}'.format(
            np.sum(f['packets'][:]['local_fifo']),
            np.sum(f['packets'][:]['shared_fifo'])))

    print('Getting good data...')
    data_mask = f['packets'][:]['packet_type'] == 0
    valid_parity_mask = f['packets'][data_mask]['valid_parity'] == 1
    good_data = (f['packets'][data_mask])[valid_parity_mask]

    print('Getting channels...')
    io_group = good_data['io_group'].astype(np.uint64)
    io_channel = good_data['io_channel'].astype(np.uint64)
    chip_id = good_data['chip_id'].astype(np.uint64)
    channel_id = good_data['channel_id'].astype(np.uint64)
    unique_channels = set(larpix_plot.unique_channel_id(io_group, io_channel, chip_id, channel_id))

    print('Loop over {} packets:'.format(len(io_group)))
    data = dict()
    for channel in unique_channels:
        print(channel,'/',list(unique_channels)[-1],end='\r')
        channel_mask = larpix_plot.unique_channel_id(io_group, io_channel, chip_id, channel_id) == channel
        timestamp = good_data[channel_mask]['timestamp']
        timestamp_s = timestamp * 100e-9

        if len(timestamp_s) <= 2:
            continue

        data[channel] = dict(
            channel_mask = channel_mask,
            timestamp = timestamp,
            timestamp_s = timestamp_s,
            rate = len(timestamp) / (max(timestamp_s) - min(timestamp_s) + 1e-9)
            )
    print()

    return data

if __name__ == '__main__':
    data = main(*sys.argv[1:])
    plot_trigger_rate(data, list(data.keys())[0])
    plot_trigger_rate_summary(data)

