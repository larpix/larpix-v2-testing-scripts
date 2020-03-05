'''
Plots test pulse as a function of injected DAC for a single chip
Use a json config file:
{
    'directory': '<path to directory>',
    'file_info': [
        ['<filename>', <dac to associate with file>, <vref_dac>, <vcm_dac>, <n pulses>]
    ],
    'pedestal_mv', [<channel_0 pedestal conv. to mv>, ...],
    'pedestal_std_mv', [<channel_0 pedestal std conv. to mv>, ...],
    'pedestal_err_mv', [<channel_0 pedestal err conv. to mv>, ...]
}

Usage:
    python3 -i plot_testpulse.py <config file>

'''
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import os
from collections import defaultdict

def adc_2_mv(adc, vref_dac, vcm_dac, vdda=1800.):
    vref_mv = vref_dac * vdda / 256.
    vcm_mv  = vcm_dac * vdda / 256.
    return adc * (vref_mv - vcm_mv) / 256. + vcm_mv

def plot_channel_eff(data, channel, name=None):
    if name is None:
        plt.figure('channel {} eff'.format(channel))
    else:
        plt.figure(name)
    x = data['tp_dacs'][channel] # tp_dac
    y = data['tp_effs'][channel] # tp_eff
    plt.plot(x, y, '.-', label='channel {}'.format(channel))
    plt.xlabel('injected pulse [DAC]')
    plt.ylabel('trigger efficiency')
    plt.legend()
    plt.tight_layout()

def plot_channel_resp(data, channel, name=None):
    if name is None:
        plt.figure('channel {} resp'.format(channel))
    else:
        plt.figure(name)
    x = data['tp_dacs'][channel] # tp_dac
    y = data['tp_ints'][channel] # tp_int
    ystd = data['tp_ints_std'][channel] # tp_int_std
    plt.errorbar(x, y, yerr=ystd, fmt='.-', label='channel {}'.format(channel))
    plt.xlabel('injected pulse [DAC]')
    plt.ylabel('integrated ADC (pedestal subtracted) [mV]')
    plt.legend()
    plt.tight_layout()

def plot_channel_peak(data, channel, name=None):
    if name is None:
        plt.figure('channel {} max trigger adc'.format(channel))
    else:
        plt.figure(name)
    x = data['tp_dacs'][channel] # tp_dac
    y = data['tp_peaks'][channel] # tp_peak
    ystd = data['tp_peaks_std'][channel]
    plt.errorbar(x, y, yerr=ystd, fmt='.-', label='channel {}'.format(channel))
    plt.xlabel('injected pulse [DAC]')
    plt.ylabel('max ADC (pedestal subtracted) [mV]')
    plt.legend()
    plt.tight_layout()

def main(*args, cluster_dt=1000000, **kwargs):
    config_filename = args[0]

    print('opening',config_filename)

    plt.ion()

    tp_dacs = defaultdict(list)
    tp_ints = defaultdict(list)
    tp_ints_std = defaultdict(list)
    tp_peaks = defaultdict(list)
    tp_peaks_std = defaultdict(list)
    tp_effs = defaultdict(list)

    with open(config_filename,'r') as fi:
        config_data = json.load(fi)
        for input_file_info in config_data['file_info']:
            directory = config_data['directory']
            filename = input_file_info[0]
            filepath = os.path.join(directory, filename)
            print('opening',filepath)
            tp_dac = input_file_info[1]
            vref_dac = input_file_info[2]
            vcm_dac = input_file_info[3]
            n_pulses = input_file_info[4]
            ped_adc = config_data['pedestal_adc']
            ped_vref_dac = config_data['pedestal_vref_dac']
            ped_vcm_dac = config_data['pedestal_vcm_dac']
            ped_mv = [adc_2_mv(ped, ped_vref_dac, ped_vcm_dac) for ped in ped_adc]

            f = h5py.File(filepath,'r')

            print('finding pulse clusters...')
            pulse_idcs = np.argwhere(f['packets'][:]['packet_type'] == 2).flatten()
            pulse_clusters = np.split(f['packets'], pulse_idcs)[1:]
            print('found',len(pulse_clusters),'clusters')

            print('collecting pulse data...')
            for channel in set(f['packets'][:]['channel_id']):
                tp_dacs[channel].append(tp_dac)

                peaks = []
                integrals = []
                n_triggers = []
                eff = 0

                for pulse in pulse_clusters:
                    if not len(pulse):
                        continue
                    channel_mask = pulse['channel_id'] == channel
                    valid_parity_mask = pulse['valid_parity'] == 1
                    packet_type_mask = pulse['packet_type'] == 0
                    mask = np.logical_and(np.logical_and(channel_mask, valid_parity_mask), packet_type_mask)
                    datawords = pulse[mask]['dataword']

                    n_triggers.append(np.sum(mask))
                    if n_triggers[-1]:
                        peaks.append(np.max(adc_2_mv(datawords, vref_dac, vcm_dac) - ped_mv[channel]))
                        integrals.append(np.sum(adc_2_mv(datawords, vref_dac, vcm_dac) - ped_mv[channel]))
                        eff += 1

                tp_peaks[channel].append(
                    np.mean(peaks)
                    )
                tp_peaks_std[channel].append(
                    np.std(peaks)/np.sqrt(len(peaks))
                    if len(peaks) > 2
                    else np.mean(peaks)
                    )
                tp_ints[channel].append(
                    np.mean(integrals)
                    )
                tp_ints_std[channel].append(
                    np.std(integrals)/np.sqrt(len(integrals))
                    if len(integrals)
                    else np.mean(integrals)
                    )
                tp_effs[channel].append(
                    eff / n_pulses)

    return dict(
        tp_dacs=tp_dacs,
        tp_ints=tp_ints,
        tp_ints_std=tp_ints_std,
        tp_effs=tp_effs,
        tp_peaks=tp_peaks,
        tp_peaks_std=tp_peaks_std
        )


if __name__ == '__main__':
    data = main(*sys.argv[1:])
