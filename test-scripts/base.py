import larpix
import larpix.io
import larpix.logger

def set_uart_clk_ctrl(controller, chip_key, clk_ctrl):
    '''
    Updates the uart clock speed for the system using the broadcast id

    clk_ctrl:
    0 = 2 clk cycles / bit
    1 = 4 clk cycles / bit
    2 = 8 clk cycles / bit
    3 = 16 clk cycles / bit

    '''
    controller[chip_key].config.clk_ctrl = clk_ctrl
    controller.write_configuration(chip_key, 123) # clk_ctrl reg
    clk_ctrl_2_clk_ratio_map = {
        0: 2,
        1: 4,
        2: 8,
        3: 16}
    c.set_larpix_uart_clk_ratio(clk_ctrl_2_clk_ratio_map[clk_ctrl])

def main(logger=False, reset=True):
    print('base config')

    # create controller
    c = larpix.Controller()
    c.io = larpix.io.SerialPort()
    if logger:
        print('logger')
        c.logger = larpix.logger.HDF5Logger(version='2.0')
        print('filename:',c.logger.filename)

    c.add_chip('1-1-1')

    if reset:
        # issues hard reset to larpix
        c.io.set_larpix_reset_cnt(128)
        c.io.larpix_reset()
        c.io.set_larpix_uart_clk_ratio(1)

    # set uart speed
    set_uart_clk_ctrl(c, '1-1-1', 1)

    # set configuration
    c['1-1-1'].config.enable_miso_upstream[0] = 1
    c['1-1-1'].config.enable_miso_downstream[0] = 1

    # write and verify
    c.write_configuration('1-1-1',[124,125])
    ok, diff = c.verify_configuration('1-1-1')
    if not ok:
        print('config error',diff)

    return c

if __name__ == '__main__':
    c = main()
