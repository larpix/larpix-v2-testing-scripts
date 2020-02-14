import larpix
import larpix.io
import larpix.logger

def main(logger=False, soft_reset=True):
    # create controller
    c = larpix.Controller()
    c.io = larpix.io.SerialPort()
    if logger:
        c.logger = larpix.logger.HDF5Logger(version='2.0')

    c.add_chip('1-1-1')

    if soft_reset:
        c['1-1-1'].config.load_config_defaults = 1
        c.write_configuration('1-1-1',[123])
        c['1-1-1'].config.load_config_defaults = 0
        c.write_configuration('1-1-1',[123])

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
