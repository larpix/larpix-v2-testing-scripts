import larpix
import larpix.io
import larpix.logger

import base
import base_warm

import argparse
import json

def main(controller_config=None, chip_key=None, threshold=128, runtime=60, channels=range(0,64,1)):
    print('rough leakage config')

    # create controller
    c = base_warm.main(controller_config, logger=True)

    chips_to_test = c.chips.keys()
    if not chip_key is None:
        chips_to_test = [chip_key]

    # set configuration
    print('threshold',threshold)
    print('channels',channels)
    for chip_key in chips_to_test:
        for channel in channels:
            c[chip_key].config.channel_mask[channel] = 0
            c[chip_key].config.threshold_global = threshold

    # write configuration
    for chip_key in chips_to_test:
        registers = list(range(131,139)) # channel mask
        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)

        registers = [64] # threshold
        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)

        ok, diff = c.verify_configuration(chip_key)
        if not ok:
            print('config error',diff)

    base.flush_data(c)
    print('run for',runtime,'sec')
    c.logger.enable()
    c.run(runtime,'collect data')
    c.logger.flush()
    print('packets read',len(c.reads[-1]))
    c.logger.disable()

    packet_channels, packet_keys = zip(*c.reads[-1].extract('channel_id','chip_key'))
    for chip_key in chips_to_test:
        for channel in channels:
            print('chip',chip_key,'channel',channel,'rate',len([ch for ch, key in zip(packet_channels, packet_keys) if ch == channel and key == chip_key])/runtime,'Hz')

    return c

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller_config', default=None, type=str)
    parser.add_argument('--chip_key', default=None, type=str)
    parser.add_argument('--threshold', default=128, type=int)
    parser.add_argument('--runtime', default=60, type=float)
    parser.add_argument('--channels', default=range(64), type=json.loads)
    args = parser.parse_args()
    c = main(**vars(args))

