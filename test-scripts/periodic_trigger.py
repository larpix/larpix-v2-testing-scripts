import larpix
import larpix.io
import larpix.logger

import base

def main(channels=range(0,64,1), rolling=True, trigger_cycles=0):
    # create controller
    c = base.main(logger=True)

    # enable digital monitor
    c['1-1-1'].config.digital_monitor_enable = 1
    c['1-1-1'].config.digital_monitor_chan = 0
    c['1-1-1'].config.digital_monitor_select = 3
    registers = [118,119]
    c.write_configuration('1-1-1',registers)

    # set configuration
    # set periodic trigger cycles first to avoid high data rates
    c['1-1-1'].config.periodic_trigger_cycles = trigger_cycles
    registers = [128] # periodic trigger cycles
    c.write_configuration('1-1-1', registers)

    c['1-1-1'].config.enable_periodic_trigger = 0
    c['1-1-1'].config.enable_rolling_periodic_trigger = rolling
    c['1-1-1'].config.enable_hit_veto = 0
    for channel in channels:
        c['1-1-1'].config.channel_mask[channel] = 0
        c['1-1-1'].config.periodic_trigger_mask[channel] = 0

    # write and verify
    registers = list(range(166,170)) # periodic trigger cycles
    c.write_configuration('1-1-1', registers)

    registers = list(range(155,163)) # periodic trigger mask
    registers += list(range(131,139)) # channel mask
    registers += [128] # periodic trigger, hit veto
    c.write_configuration('1-1-1', registers)

    c['1-1-1'].config.enable_fifo_diagnostics = 1
    larpix.Packet_v2.fifo_diagnostics_enabled = True
    registers = [123] # fifo diagnostics

    c.write_configuration('1-1-1', registers)
    ok, diff = c.verify_configuration('1-1-1')
    if not ok:
        print('config error',diff)

    c.logger.enable()

    # write config and listen for 2s
    c['1-1-1'].config.enable_periodic_trigger = 1
    c.write_configuration('1-1-1', 128, write_read=2)
    print('read for 2s:',len(c.reads[-1]))

    c.logger.flush()
    print('filename:',c.logger.filename)

    c.logger.disable()

    return c

if __name__ == '__main__':
    c = main()
