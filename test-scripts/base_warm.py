import larpix
import larpix.io
import larpix.logger

import base

def main(*args, **kwargs):
    print('base warm config')

    # create controller
    c = base.main(*args, **kwargs)

    # set configuration
    c['1-1-1'].config.ibias_buffer = 3
    c['1-1-1'].config.ibias_tdac = 5
    c['1-1-1'].config.ibias_comp = 5
    c['1-1-1'].config.ibias_csa = 7

    c['1-1-1'].config.ref_current_trim = 15

    c['1-1-1'].config.vref_dac = 185
    c['1-1-1'].config.vcm_dac = 41

    # write configuration
    registers = [74, 75, 76, 77] # ibias
    registers += [81] # ref current
    registers += [82, 83] # vXX_dac
    c.write_configuration('1-1-1',registers)

    # verify
    ok, diff = c.verify_configuration('1-1-1')
    if not ok:
        print('config error',diff)

    return c

if __name__ == '__main__':
    c = main()
