import larpix
import larpix.io
import larpix.logger

import base

def main(channel0=0, channel1=1, runtime=12):
    # create controller
    c = base.main(logger=True)

    # set configuration
    c['1-1-1'].config.external_trigger_mask[channel0] = 0
    c['1-1-1'].config.external_trigger_mask[channel1] = 0
    c['1-1-1'].config.channel_mask[channel0] = 0
    c['1-1-1'].config.channel_mask[channel1] = 0
    c['1-1-1'].config.enable_hit_veto = 0
    c['1-1-1'].config.csa_bypass_enable = 1
    c['1-1-1'].config.csa_bypass_select[channel0] = 1
    c['1-1-1'].config.csa_bypass_select[channel1] = 1

    # write and verify
    c.write_configuration('1-1-1')
    ok, diff = c.verify_configuration('1-1-1')
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
    c = main()
