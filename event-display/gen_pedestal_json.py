import argparse
import time
import json
import h5py
import numpy as np

import larpix

_default_vdda = 1800
_default_vref_dac = 217
_default_vcm_dac = 71
_default_mean_trunc = 3

def key2unique(key, channel):
    io_group,io_channel,chip_id = str(key).split('-')
    return ((int(io_group)*256 + int(io_channel))*256 + int(chip_id))*64 + int(channel)

def adc2mv(adc, ref, cm, bits=8):
    return (ref-cm) * adc/(2**bits) + cm

def dac2mv(dac, max, bits=8):
    return max * dac/(2**bits)

def main(controller_config, infile, vdda=_default_vdda, vref_dac=_default_vref_dac,
    vcm_dac=_default_vcm_dac, mean_trunc=_default_mean_trunc, **kwargs):
    c = larpix.Controller()
    c.load(controller_config)

    f = h5py.File(infile,'r')
    good_data_mask = f['packets']['packet_type'] == 0
    good_data_mask = np.logical_and(f['packets']['valid_parity'] == 1, good_data_mask)

    unique_id = ((f['packets'][good_data_mask]['io_group']*256 \
        + f['packets'][good_data_mask]['io_channel'])*256 \
        + f['packets'][good_data_mask]['chip_id'])*64 \
        + f['packets'][good_data_mask]['channel_id']

    counter = 0
    total = len(c.chips)*64
    now = time.time()
    config_dict = dict()
    for chip_key in c.chips:
        for channel in range(64):
            counter += 1
            if time.time() > now + 1:
                print('{}/{}\r'.format(counter,total),end='')
                now = time.time()
            unique = key2unique(chip_key,channel)
            vref_mv = dac2mv(vref_dac,vdda)
            vcm_mv = dac2mv(vcm_dac,vdda)

            channel_mask = unique_id == unique
            adcs = f['packets']['dataword'][good_data_mask][channel_mask]
            if len(adcs) < 1:
                continue
            vals,bins = np.histogram(adcs,bins=np.arange(256))
            peak_bin = np.argmax(vals)
            min_idx,max_idx = np.max(peak_bin-mean_trunc,0), np.min(peak_bin+mean_trunc,len(vals)-1)
            ped_adc = np.average(bins[min_idx:max_idx]+0.5, weights=vals[min_idx:max_idx])

            config_dict[unique] = dict(
                pedestal_mv=adc2mv(ped_adc,vref_mv,vcm_mv)
                )
    with open(infile.strip('.h5')+'evd_ped.json','w') as fo:
        json.dump(config_dict, fo, sort_keys=True, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
        A script for generating pedestal configurations used by the to_evd_file.py
        script. To use, specify the ``controller_config`` that was used for the
        pedestal run, the path to the pedestal datafile, and the settings used
        for the pedestal run (vdda,vref,vcm). This script will then take the
        truncated mean for each channel's adc values and store them in pedestal
        config file.'''
        )
    parser.add_argument('--controller_config','-c',required=True,type=str)
    parser.add_argument('--infile','-i',required=True,type=str)
    parser.add_argument('--vdda',default=_default_vdda,type=float,help='''default=%(default)s mV''')
    parser.add_argument('--vref_dac',default=_default_vref_dac,type=int,help='''default=%(default)s''')
    parser.add_argument('--vcm_dac',default=_default_vcm_dac,type=int,help='''default=%(default)s''')
    parser.add_argument('--mean_trunc',default=_default_mean_trunc,type=int,help='''
        default=%(default)s''')
    args = parser.parse_args()
    main(**vars(args))
