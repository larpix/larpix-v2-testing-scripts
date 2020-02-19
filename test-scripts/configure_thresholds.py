import larpix
import larpix.io
import larpix.logger

import base
import base_warm

def main(channels=range(0,64,1), threshold_global_start=40, pixel_trim_dac_start=31, runtime=1, target_rate=1):
    print('configure thresholds')

    # create controller
    c = base_warm.main()

    # set initial configuration
    print('channels',channels)
    print('threshold_global_start', threshold_global_start)
    print('pixel_trim_dac_start', pixel_trim_dac_start)

    for channel in channels:
        c['1-1-1'].config.channel_mask[channel] = 0
        c['1-1-1'].config.pixel_trim_dac = pixel_trim_dac_start
    c['1-1-1'].config.threshold_global = threshold_global_start

    c['1-1-1'].config.enable_periodic_reset = True

    # write configuration
    registers = list(range(131,139)) # channel mask
    c.write_configuration('1-1-1', registers)
    registers = [128] # periodic reset
    c.write_configuration('1-1-1', registers)
    registers = list(range(0,65)) # trim and threshold
    c.write_configuration('1-1-1', registers)

    # verify
    ok, diff = c.verify_configuration('1-1-1')
    if not ok:
        print('config error',diff)

    # walk down global threshold
    for threshold in range(threshold_global_start,0,-1):
        # update threshold
        c['1-1-1'].config.threshold_global = threshold
        registers = [64]
        c.write_configuration('1-1-1', registers)
        print('threshold_global',threshold)

        # check rate
        c.run(runtime,'rate check')
        rate = len(c.reads[-1])/runtime
        print('rate',rate)

        # stop if rate is above target
        if rate > target_rate:
            # back off by 1
            c['1-1-1'].config.threshold_global = threshold+1
            registers = [64]
            c.write_configuration('1-1-1', registers)
            break

    c.run(1,'flush stale data')

    # walk down pixel trims
    channels_to_set = set(channels)
    trim = pixel_trim_dac_start
    while channels_to_set != set():
        # update channel trims
        for channel in channels_to_set:
            if not c['1-1-1'].config.pixel_trim_dac[channel] == 0:
                c['1-1-1'].config.pixel_trim_dac[channel] = trim
            else:
                # remove channels that have bottomed out
                channels_to_set.remove(channel)
        registers = list(range(0,64))
        c.write_configuration('1-1-1', registers)
        print('pixel_trim_dac',trim)
        print('channels_to_set',channels_to_set)

        # check rate
        c.run(runtime,'rate check')
        rate = len(c.reads[-1])/runtime
        print('rate',rate)

        # back off channels that have rates above target
        channel_triggers = c.reads[-1].extract('channel_id')
        rates = dict([(channel, len([ch for ch in channel_triggers if ch == channel])/runtime) for channel in channels_to_set])
        for channel in channels_to_set:
            if rates[channel] > target_rate:
                # back off by 1
                c['1-1-1'].config.pixel_trim_dac[channel] = trim+1
                if channel in channels_to_set:
                    channels_to_set.remove(channel)
        registers = list(range(0,64))
        c.write_configuration('1-1-1', registers)

        # walk down trim if no channels above rate
        if none([rate > target_rate for rate in rates]):
            trim -= 1

    c.run(1,'flush stale data')

    # save config
    time_format = '%Y_%m_%d_%H_%M_%S_%Z'
    config_filename = 'config-'+time.strftime(time_format)+'.json'
    c['1-1-1'].config.write(filename, force=True)
    print('saved to',config_filename)

    return c

if __name__ == '__main__':
    c = main()
