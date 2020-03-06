import sys
import time

import larpix
import larpix.io
import larpix.logger

clk_ctrl_2_clk_ratio_map = {
        0: 2,
        1: 4,
        2: 8,
        3: 16
        }

def flush_data(controller, runtime=0.1, rate_limit=0., max_iterations=10):
    '''
    Continues to read data until data rate is less than rate_limit

    '''
    for _ in range(max_iterations):
        controller.run(runtime, 'flush_data')
        if len(controller.reads[-1])/runtime < rate_limit:
            break

def main(controller_config_file=None, logger=False, reset=True):
    print('base config')

    # create controller
    c = larpix.Controller()
    c.io = larpix.io.SerialPort()
    if logger:
        print('logger')
        c.logger = larpix.logger.HDF5Logger(version='2.0')
        print('filename:',c.logger.filename)

    if controller_config_file is None:
        c.add_chip('1-1-1')
        # create node for fpga
        c.add_network_node(1, 1, c.network_names, 'ext', root=True)
        # create upstream link on uart 0
        c.add_network_link(1,1,'miso_us',('ext',1),0)
        # create downstream link on uart 0
        c.add_network_link(1,1,'miso_ds',(1,'ext'),0)
        # create mosi link
        c.add_network_link(1,1,'mosi',('ext',1),0)
    else:
        c.load(controller_config_file)

    if reset:
        # issues hard reset to larpix
        c.io.set_larpix_uart_clk_ratio(clk_ctrl_2_clk_ratio_map[0])
        c.io.set_larpix_reset_cnt(128)
        c.io.larpix_reset()

    # initialize network
    for io_group, io_channels in c.network.items():
        for io_channel in io_channels:
            c.init_network(io_group, io_channel)
            c.init_network(io_group, io_channel)
            ok, diff = c.verify_network(c.get_network_keys(io_group, io_channel))
            if not ok:
                print('network',io_group,io_channel,'config error',diff)
            else:
                print('network',io_group,io_channel,'configured ok')

    # set uart speed
    clk_ctrl = 1
    for io_group, io_channels in c.network.items():
        for io_channel in io_channels:
            chip_keys = c.get_network_keys(io_group,io_channel,root_first_traversal=False)
            for chip_key in chip_keys:
                c[chip_key].config.clk_ctrl = clk_ctrl
                c.write_configuration(chip_key, 'clk_ctrl')
                c.write_configuration(chip_key, 'clk_ctrl')
    c.io.set_larpix_uart_clk_ratio(clk_ctrl_2_clk_ratio_map[clk_ctrl])

    # set other configuration registers
    for chip_key in c.chips:
        registers = []
        register_map = c[chip_key].config.register_map

        c[chip_key].config.csa_gain = 1
        registers += list(register_map['csa_gain'])
        c[chip_key].config.adc_hold_delay = 15
        registers += list(register_map['adc_hold_delay'])
        c[chip_key].config.enable_miso_differential = [1,1,1,1]
        registers += list(register_map['enable_miso_differential'])

        c.write_configuration(chip_key, registers)
        c.write_configuration(chip_key, registers)
    # verify
    for chip_key in c.chips:
        ok, diff = c.verify_configuration(chip_key)
        if not ok:
            print('config error',diff)
        else:
            print('configured ok')

    return c

if __name__ == '__main__':
    c = main(*sys.argv[1:])
