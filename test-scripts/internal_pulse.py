'''
Loads specified configuration file and issues a set number of test pulses to the specified channels

Usage:
  python3 -i internal_pulse.py <config file>

'''
import larpix
import larpix.io
import larpix.logger

import base_warm
import load_config

import sys

def main(config_name=None, controller_config=None, pulse_dac=10, n_pulses=10, channels=range(0,64,1), runtime=0.01):
    print('configure thresholds')
    pulse_dac = int(pulse_dac)
    n_pulses = int(n_pulses)

    # create controller
    c = None
    if config_name is None:
        c = base_warm.main(controller_config, logger=True)
    else:
        if controller_config is None or controller_config == 'None':
            c = load_config.main(config_name, logger=True)
        else:
            c = load_config.main(config_name, controller_config, logger=True)

    # set initial configuration
    print('channels',channels)
    print('pulse_dac', pulse_dac)
    print('n_pulses', n_pulses)

    for chip, chip_key in c.chips.items():
        chip.config.adc_hold_delay = 15
        registers = [129]
        c.write_configuration(chip_key, registers)

        c.enable_testpulse(chip_key, channels)

    # verify
    ok, diff = c.verify_configuration()
    if not ok:
        print('config error',diff)
    else:
        print('config ok')

    # add a dummy for chip 255
    for io_group, io_channels in c.network.items():
        for io_channel in io_channels:
            c.add_chip(Key(io_group,io_channel,255))

    # issue pulses
    for i in range(n_pulses):
        for io_group, io_channels in c.network.items():
            for io_channel in io_channels:
                c.logger.enable()
                c.issue_testpulse(Key(io_group,io_channel,255),pulse_dac,read_time=runtime)
                c.logger.disable()

                print('pulse',i,'triggers',len(c.reads[-1]))

                c.enable_testpulse(Key(io_group,io_channel,255),channels)
                c.run(runtime,'flush stale data')

    return c

if __name__ == '__main__':
    c = main(*sys.argv[1:])
