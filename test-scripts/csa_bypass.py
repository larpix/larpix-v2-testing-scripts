import sys

import larpix
import larpix.io
import larpix.logger

import base

def main(controller_config=None, channel=0, runtime=12):
    print('csa bypass')

    # create controller
    c = base.main(controller_config_file=controller_config, logger=True)

    # set configuration
    print('channel',channel)
    for chip_key, chip in c.chips.items():
        chip.config.external_trigger_mask[channel] = 0
        chip.config.channel_mask[channel] = 0
        chip.config.enable_hit_veto = 0
        chip.config.csa_bypass_enable = 1
        chip.config.csa_bypass_select[channel] = 1

        registers = list()
        registers += chip.config.register_map['external_trigger_mask']
        registers += chip.config.register_map['channel_mask']
        registers += chip.config.register_map['enable_hit_veto']
        registers += chip.config.register_map['csa_bypass_enable']
        registers += chip.config.register_map['csa_bypass_select']

        # write
        c.write_configuration(chip_key, registers)

    # verify
    ok, diff = c.verify_configuration()
    if not ok:
        print('config error',diff)

    # take data
    print('taking test data...')
    c.run(0.5,'test')
    print(c.reads[-1])
    print('received packets:',len(c.reads[-1]))

    print('taking full data...')
    print('file: ', c.logger.filename)
    c.logger.enable()
    c.run(runtime,'data')
    print('received packets:',len(c.reads[-1]))
    c.logger.flush()
    c.logger.disable()

    return c

if __name__ == '__main__':
    c = main(*sys.argv[1:])
