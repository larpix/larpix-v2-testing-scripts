import sys

import larpix
import larpix.io

import base
import base_warm

def main(controller_config=None):
    # create controller
    c = base_warm.main(controller_config_file=controller_config, logger=True)

    # set configuration
    for chip_key, chip in c.chips.items():
        chip.config.periodic_trigger_mask = [0]*64
        chip.config.channel_mask = [0]*64
        chip.config.periodic_trigger_cycles = 100000 # 100Hz
        chip.config.enable_periodic_trigger = True
        chip.config.enable_rolling_periodic_trigger = True
        chip.config.enable_periodic_reset = True
        chip.config.enable_hit_veto = False

        # write configuration
        registers = list(range(155,163)) # periodic trigger mask
        registers += list(range(131,139)) # channel mask
        registers += list(range(166,170)) # periodic trigger cycles
        registers += [128] # periodic trigger, reset, rolling trigger, hit veto

        c.write_configuration(chip_key, registers)

    ok, diff = c.verify_configuration()
    if not ok:
        print('config error',diff)

    c.logger.enable()
    c.run(10,'collect data')
    c.logger.flush()
    print('packets read',len(c.reads[-1]))
    c.logger.disable()

    return c

if __name__ == '__main__':
    c = main(*sys.argv[1:])
