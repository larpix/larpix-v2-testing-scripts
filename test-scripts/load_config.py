'''
Creates a base controller object and loads the specified configuration onto the chip

Usage:
    python3 -i load_config.py <configuration name>

'''

import sys

import larpix
import larpix.io
import larpix.logger

import base

def main(config_name, *args, **kwargs):
    print('load config')

    # create controller
    c = base.main(*args, **kwargs)

    # set configuration
    c['1-1-1'].config.load(config_name)

    # write configuration
    c.write_configuration('1-1-1')

    # verify
    ok, diff = c.verify_configuration('1-1-1')
    if not ok:
        print('config error',diff)

    return c

if __name__ == '__main__':
    c = main(*sys.argv[1:])
