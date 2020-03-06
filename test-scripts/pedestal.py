import sys
import argparse

import larpix
import larpix.io

import base
import base_warm

def main(controller_config=None, periodic_trigger_cycles=100000, runtime=10, *args, **kwargs):
    # create controller
    c = base_warm.main(controller_config_file=controller_config, logger=True)

    # set configuration
    for chip_key, chip in c.chips.items():
        chip.config.periodic_trigger_mask = [0]*64
        chip.config.channel_mask = [0]*64
        chip.config.periodic_trigger_cycles = periodic_trigger_cycles
        chip.config.enable_periodic_trigger = 1
        chip.config.enable_rolling_periodic_trigger = 1
        chip.config.enable_periodic_reset = 1
        chip.config.enable_hit_veto = 0

        # write configuration
        registers = list(range(155,163)) # periodic trigger mask
        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)
        registers = list(range(131,139)) # channel mask
        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)
        registers = list(range(166,170)) # periodic trigger cycles
        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)
        registers = [128] # periodic trigger, reset, rolling trigger, hit veto
        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)

    for chip_key in c.chips:
        ok, diff = c.verify_configuration(chip_key)
        if not ok:
            print('config error',diff)

    base.flush_data(c, rate_limit=(1+1/(periodic_trigger_cycles*1e-7)*len(c.chips)))
    c.logger.enable()
    c.run(runtime,'collect data')
    c.logger.flush()
    print('packets read',len(c.reads[-1]))
    c.logger.disable()

    return c

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller_config', default=None, type=str)
    parser.add_argument('--periodic_trigger_cycles', default=100000, type=int)
    parser.add_argument('--runtime', default=10, type=float)
    args = parser.parse_args()
    c = main(**vars(args))

