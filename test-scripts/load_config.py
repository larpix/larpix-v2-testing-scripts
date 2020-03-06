'''
Creates a base controller object and loads the specified configuration onto the chip

Usage:
    python3 -i load_config.py <configuration name>

'''

import sys
import os
import glob

import larpix
import larpix.io
import larpix.logger

import base

config_format = 'config-{chip_key}-*.json$'

def main(config_name, *args, **kwargs):
    print('load config')

    # create controller
    c = base.main(*args, **kwargs)

    # set configuration
    if not os.isdir(config_name):
        for chip_key,chip in c.chips.items():
            chip.config.load(config_name)
    else:
        # load all configurations for chips
        for chip_key,chip in c.chips.items():
            config_files = glob.glob(os.join(config_name, config_format.format(chip_key=chip_key)))
            if config_files:
                chip.config.load(sorted(config_files[-1]))

    # write configuration
    for chip_key, chip in c.chips.items():
        c.write_configuration(chip_key)

    # verify
    ok, diff = c.verify_configuration()
    if not ok:
        print('config error',diff)

    return c

if __name__ == '__main__':
    c = main(*sys.argv[1:])
