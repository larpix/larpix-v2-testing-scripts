'''
Loads specified configuration file and issues a set number of test pulses to the specified channels

Usage:
  python3 -i internal_pulse.py <config file>

'''
import larpix
import larpix.io
import larpix.logger

import base
import base_warm
import load_config

import argparse
import json

def main(config_name=None, controller_config=None, chip_key=None, pulse_dac=10, n_pulses=10, channels=range(0,64,1), runtime=0.01):
    print('configure thresholds')
    pulse_dac = int(pulse_dac)
    n_pulses = int(n_pulses)

    # create controller
    c = None
    if config_name is None:
        c = base_warm.main(controller_config, logger=True)
    else:
        if controller_config is None:
            c = load_config.main(config_name, logger=True)
        else:
            c = load_config.main(config_name, controller_config, logger=True)

    # set initial configuration
    print('channels',channels)
    print('pulse_dac', pulse_dac)
    print('n_pulses', n_pulses)

    chips_to_test = c.chips.keys()
    if not chip_key is None:
        chips_to_test = [chip_key]

    for chip_key in chips_to_test:
        c[chip_key].config.adc_hold_delay = 15
        registers = [129]
        c.write_configuration(chip_key, registers)
        c.enable_testpulse(chip_key, channels)

    # verify
    for chip_key in chips_to_test:
        ok, diff = c.verify_configuration(chip_key)
        if not ok:
            print('config error',diff)
        else:
            print('config ok')

    # issue pulses
    base.flush_data(c, rate_limit=len(c.chips)*64)
    for i in range(n_pulses):
        for chip_key in chips_to_test:
            c.logger.enable()
            c.issue_testpulse(chip_key, pulse_dac, read_time=runtime)
            c.logger.disable()

            print('pulse',i,chip_key,'triggers',len(c.reads[-1]))

            c.enable_testpulse(chip_key, channels)
            base.flush_data(c, rate_limit=len(c.chips)*64)

    return c

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=None, type=str)
    parser.add_argument('--controller_config', default=None, type=str)
    parser.add_argument('--chip_key', default=None, type=str)
    parser.add_argument('--pulse_dac', default=10, type=int)
    parser.add_argument('--n_pulses', default=10, type=int)
    parser.add_argument('--channels', default=range(64), type=json.loads)
    parser.add_argument('--runtime', default=0.1, type=float)
    args = parser.parse_args()
    c = main(**vars(args))
