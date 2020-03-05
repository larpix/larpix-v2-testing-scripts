import larpix
import larpix.io
import larpix.logger

import base

def main(*args, **kwargs):
    print('base warm config')

    # create controller
    c = base.main(*args, **kwargs)

    # set configuration
    for chip_key, chip in c.chips.items():
        chip.config.ibias_buffer = 3
        chip.config.ibias_tdac = 5
        chip.config.ibias_comp = 5
        chip.config.ibias_csa = 7

        chip.config.ref_current_trim = 15

        chip.config.vref_dac = 185
        chip.config.vcm_dac = 41

        # write configuration
        registers = [74, 75, 76, 77] # ibias
        registers += [81] # ref current
        registers += [82, 83] # vXX_dac

        c.write_configuration(chip_key, registers)

    # verify
    ok, diff = c.verify_configuration()
    if not ok:
        print('config error',diff)

    return c

if __name__ == '__main__':
    c = main()
