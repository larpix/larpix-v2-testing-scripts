'''
Plots ADC vs. timestamp, ADC vs. packet index, and timestamp vs. packet index

Usage:
  python3 -i plot_dual_csa_bypass.py <filename>

'''
import h5py
import matplotlib.pyplot as plt
import sys

def main(*args):
    filename = args[0]
    print('opening',filename)

    plt.ion()

    f = h5py.File(filename,'r')

    unique_channel = f['packets'][:]['channel_id'].astype(int) + 100*f['packets'][:]['chip_id'].astype(int)
    channels = set(unique_channel)
    # valid_parity_mask = f['packets'][:]['valid_parity'] == 1

    data = dict()
    for channel in channels:
        channel_mask = (unique_channel == channel)
        packet_idx = [i for i,ch in enumerate(unique_channel) if ch == channel]
        timestamp = f['packets'][channel_mask]['timestamp']
        adc = f['packets'][channel_mask]['dataword']

        data[channel] = dict(
            channel_mask = channel_mask,
            packet_idx = packet_idx,
            timestamp = timestamp,
            adc = adc
            )

    alpha=0.5

    plt.figure('timestamp v. index')
    for channel in data.keys():
        plt.plot(
            data[channel]['packet_idx'],
            data[channel]['timestamp'],
            '.',
            alpha=alpha
            )
    plt.legend(data.keys())

    plt.figure('adc v. index')
    for channel in data.keys():
        plt.plot(
            data[channel]['packet_idx'],
            data[channel]['adc'],
            '.',
            alpha=alpha
            )
    plt.legend(data.keys())

    plt.figure('adc v. timestamp')
    for channel in data.keys():
        plt.plot(
            data[channel]['timestamp'],
            data[channel]['adc'],
            '.',
            alpha=alpha
            )
    plt.legend(data.keys())

    plt.figure('channel v. timestamp')
    for channel in data.keys():
        plt.plot(
            data[channel]['timestamp'],
            [channel for value in data[channel]['timestamp']],
            '.',
            alpha=alpha
            )
    plt.legend(data.keys())

    return f

if __name__ == '__main__':
    f = main(*sys.argv[1:])
