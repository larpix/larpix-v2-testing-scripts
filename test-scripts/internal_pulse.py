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

def main(config_name=None, channels=range(0,64,1), pulse_dac=10, n_pulses=10):
    print('configure thresholds')

    # create controller
    c = None
    if config_name is None:
        c = base_warm.main(logger=True)
    else:
        c = load_config.main(config_name, logger=True)

    # set initial configuration
    print('channels',channels)
    print('pulse_dac', pulse_dac)
    print('n_pulses', n_pulses)

    c.['1-1-1'].config.adc_hold_delay = 15
    registers = [129]
    c.write_configuration('1-1-1', registers)

    c.enable_testpulse('1-1-1',channels)

    # verify
    ok, diff = c.verify_configuration('1-1-1')
    if not ok:
        print('config error',diff)

    # issue pulses
    for i in range(n_pulses):
        c.logger.enable()
        c.issue_testpulse(pulse_dac)
        c.logger.disable()

        print('pulse',i,'triggers',len(c.reads[-1]))

        c.enable_testpulse('1-1-1',channels)
        c.run(0.1,'flush stale data')

    return c

if __name__ == '__main__':
    c = main(*sys.argv[1:])
