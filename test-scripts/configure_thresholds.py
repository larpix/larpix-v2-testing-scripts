import larpix
import larpix.io
import larpix.logger

import base
import base_warm

import time
import sys
from copy import copy
import argparse
import json

def main(controller_config=None, chip_key=None, threshold_global_start=30, channels=range(0,64,1),  pixel_trim_dac_start=31, runtime=1, target_rate=2):
    print('configure thresholds')

    # create controller
    c = base_warm.main(controller_config_file=controller_config)

    # set initial configuration
    print('channels',channels)
    print('threshold_global_start', threshold_global_start)
    print('pixel_trim_dac_start', pixel_trim_dac_start)

    chips_to_configure = c.chips
    if not chip_key is None:
        chips_to_configure = [chip_key]
    for chip_key in chips_to_configure:
        print('chip',chip_key)
        for channel in channels:
            c[chip_key].config.channel_mask[channel] = 0
            c[chip_key].config.pixel_trim_dac[channel] = pixel_trim_dac_start
        c[chip_key].config.threshold_global = threshold_global_start

        c[chip_key].config.enable_periodic_reset = 1

        # write configuration
        registers = list(range(131,139)) # channel mask
        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)
        registers = [128] # periodic reset
        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)
        registers = list(range(0,65)) # trim and threshold
        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)

        # verify
        ok, diff = c.verify_configuration(chip_key)
        if not ok:
            print('config error',diff)

        # walk down global threshold
        for threshold in range(threshold_global_start,0,-1):
            # update threshold
            c[chip_key].config.threshold_global = threshold
            registers = [64]
            c.write_configuration(chip_key, registers)
            c.write_configuration(chip_key, registers)
            print('threshold_global',threshold)

            # check rate
            base.flush_data(c)
            c.run(runtime,'rate check')
            rate = len(c.reads[-1].extract('channel_id',chip_key=chip_key))/runtime
            print('rate',rate)

            # stop if rate is above target
            if rate > target_rate:
                # back off by 1
                c[chip_key].config.threshold_global = min(threshold+1,255)
                registers = [64]
                c.write_configuration(chip_key, registers)
                c.write_configuration(chip_key, registers)
                break

        c.run(1,'flush stale data')

        # walk down pixel trims
        channels_to_set = set(channels)
        trim = pixel_trim_dac_start
        while channels_to_set != set():
            # update channel trims
            channels_to_set_copy = copy(channels_to_set)
            for channel in channels_to_set_copy:
                if not c[chip_key].config.pixel_trim_dac[channel] == 0:
                    c[chip_key].config.pixel_trim_dac[channel] = trim
                else:
                    # remove channels that have bottomed out
                    channels_to_set.remove(channel)
            registers = list(range(0,64))
            c.write_configuration(chip_key, registers)
            c.write_configuration(chip_key, registers)
            print('pixel_trim_dac',trim)
            print('channels_to_set',channels_to_set)

            # check rate
            base.flush_data(c)
            c.run(runtime,'rate check')
            rate = len(c.reads[-1])/runtime
            print('rate',rate)

            # back off channels that have rates above target
            channel_triggers = c.reads[-1].extract('channel_id',chip_key=chip_key)
            rates = dict([(channel, len([ch for ch in channel_triggers if ch == channel])/runtime) for channel in channels_to_set])
            print(rates,'rates')
            channels_to_set_copy = copy(channels_to_set)
            for channel in channels_to_set_copy:
                if rates[channel] > target_rate:
                    # back off by 1
                    c[chip_key].config.pixel_trim_dac[channel] = min(trim+1,31)
                    if channel in channels_to_set:
                        channels_to_set.remove(channel)
            registers = list(range(0,64))
            c.write_configuration(chip_key, registers)
            c.write_configuration(chip_key, registers)

            # walk down trim if no channels above rate
            if not any([rate > target_rate for rate in rates.values()]):
                trim -= 1

        base.flush_data(c)

        # save config
        time_format = '%Y_%m_%d_%H_%M_%S_%Z'
        config_filename = 'config-'+str(chip_key)+'-'+time.strftime(time_format)+'.json'
        c[chip_key].config.write(config_filename, force=True)
        print('saved to',config_filename)

        c.reads = [] # reset reads array

    return c

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller_config', default=None, type=str)
    parser.add_argument('--chip_key', default=None, type=str)
    parser.add_argument('--threshold_global_start', default=30, type=int)
    parser.add_argument('--channels', default=range(64), type=json.loads)
    parser.add_argument('--pixel_trim_dac_start', default=31, type=int)
    parser.add_argument('--runtime', default=1, type=float)
    parser.add_argument('--target_rate', default=2, type=float)
    args = parser.parse_args()
    c = main(**vars(args))
