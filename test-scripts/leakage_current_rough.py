import larpix
import larpix.io
import larpix.logger

import base
import base_warm

def main(threshold=128, runtime=60, channels=range(0,64,1)):
    print('rough leakage config')

    # create controller
    c = base_warm.main(logger=True)

    # set configuration
    print('channels',channels)
    for channel in channels:
        c['1-1-1'].config.channel_mask[channel] = 0
    c['1-1-1'].config.threshold_global = threshold
    print('threshold',threshold)

    # write configuration
    registers = list(range(131,139)) # channel mask
    c.write_configuration('1-1-1', registers)

    registers = [64] # threshold
    c.write_configuration('1-1-1', registers)

    ok, diff = c.verify_configuration('1-1-1')
    if not ok:
        print('config error',diff)

    print('run for',runtime,'sec')
    c.logger.enable()
    c.run(runtime,'collect data')
    c.logger.flush()
    print('packets read',len(c.reads[-1]))
    c.logger.disable()

    packet_channels = c.reads[-1].extract('channel_id')
    for channel in channels:
        print('channel',channel,'rate',len([ch for ch in packet_channels if ch == channel])/runtime,'Hz')

    return c

if __name__ == '__main__':
    c = main()
