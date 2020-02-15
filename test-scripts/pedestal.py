import larpix
import larpix.io

import base
import base_warm

def main():
    # create controller
    c = base_warm.main(logger=True)

    # set configuration
    c['1-1-1'].config.periodic_trigger_mask = [0]*64
    c['1-1-1'].config.channel_mask = [0]*64
    c['1-1-1'].config.periodic_trigger_cycles = 100000 # 100Hz
    c['1-1-1'].config.enable_periodic_trigger = True
    c['1-1-1'].config.enable_rolling_periodic_trigger = True
    c['1-1-1'].config.enable_periodic_reset = True
    c['1-1-1'].config.enable_hit_veto = False

    # write configuration
    registers = list(range(155,163)) # periodic trigger mask
    registers += list(range(131,139)) # channel mask
    registers += list(range(166,170)) # periodic trigger cycles
    registers += [128] # periodic trigger, reset, rolling trigger, hit veto
    c.write_configuration('1-1-1', registers)

    ok, diff = c.verify_configuration('1-1-1')
    if not ok:
        print('config error',diff)

    c.logger.enable()
    c.run(10,'collect data')
    c.logger.flush()
    print('packets read',len(c.reads[-1]))
    c.logger.disable()

    return c

if __name__ == '__main__':
    c = main()
