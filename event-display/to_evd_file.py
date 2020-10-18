import numpy as np
import argparse

from evd_lib import *

_default_geometry_file      = None
_default_configuration_file = None
_default_pedestal_file      = None
_default_buffer_size        = 38400
_default_event_dt           = 1500
_default_nhit_cut           = 2
_default_max_packets        = -1
_default_dbscan_eps         = 14.
_default_dbscan_min_samples = 5
_default_vd                 = 1.648
_default_clock_period       = 0.1

def main(in_filename, out_filename, *args,
         configuration_file=_default_configuration_file, geometry_file=_default_geometry_file,
         pedestal_file=_default_pedestal_file,
         buffer_size=_default_buffer_size, event_dt=_default_event_dt, nhit_cut=_default_nhit_cut,
         max_packets=_default_max_packets,
         dbscan_eps=_default_dbscan_eps, dbscan_min_samples=_default_dbscan_min_samples,
         vd=_default_vd, clock_period=_default_clock_period,
         **kwargs):
    # load larpix file
    larpix_logfile = load_larpix_logfile(in_filename)
    packets        = larpix_logfile['packets']

    # create buffered output file
    evd_file       = LArPixEVDFile(
        out_filename,
        source_file        = in_filename,
        geometry_file      = geometry_file,
        configuration_file = configuration_file,
        pedestal_file      = pedestal_file,
        builder_config     = dict(
            buffer_size        = buffer_size,
            event_dt           = event_dt,
            nhit_cut           = nhit_cut,
            max_packets        = max_packets
            ),
        fitter_config      = dict(
            vd                 = vd,
            clock_period       = clock_period,
            dbscan_eps         = dbscan_eps,
            dbscan_min_samples = dbscan_min_samples
            )
        )
    event_counter  = 0
    packet_counter = 0

    # remove configuration messages and packets with bad parity
    good_parity_mask = packets[:]['valid_parity']
    data_packet_mask = packets[:]['packet_type'] == 0
    mask             = np.logical_and(good_parity_mask, data_packet_mask)
    n_packets        = int(np.sum(mask))

    start_idx    = 0
    end_idx      = buffer_size
    event_buffer = np.array([])
    while start_idx <= n_packets and (max_packets < 0 or start_idx <= max_packets):
        # load a buffer of data
        packet_buffer = packets[mask][start_idx:min(end_idx,n_packets)]

        # sort packets to fix 512 bug
        packet_buffer = np.sort(packet_buffer, order='timestamp')

        # cluster into events by delta t
        packet_dt = packet_buffer[1:]['timestamp'].astype(int) - packet_buffer[:-1]['timestamp'].astype(int)
        if len(event_buffer):
            packet_dt = np.insert(packet_dt, [0], packet_buffer[0]['timestamp'].astype(int) - event_buffer[-1]['timestamp'].astype(int))
        event_idx = np.argwhere(packet_dt > event_dt).flatten() + 1
        events    = np.split(packet_buffer, event_idx)
        for idx, event in zip(event_idx, events[:-1]):
            # if len(event) >= nhit_cut or len(event_buffer) >= nhit_cut:
            #     evd_file._fit_tracks([event], plot=True)
            if len(event) >= nhit_cut:
                if idx == 0 and len(event_buffer):
                    # current event buffer is a complete event
                    evd_file.append(event_buffer)
                    event_counter  += 1
                elif len(event_buffer):
                    # event found within packet buffer (combine with existing event buffer)
                    event = np.append(event_buffer, event)
                    evd_file.append(event)
                    event_counter  += 1
                    event_buffer = np.array([])
                else:
                    evd_file.append(event)
                    event_counter  += 1
            elif len(event_buffer) >= nhit_cut:
                # current event buffer is a complete event
                evd_file.append(event_buffer)
                event_counter  += 1
                event_buffer = np.array([])
        # keep any lingering packets for next iteration
        event_buffer = np.array(events[-1])

        # increment buffer indices
        start_idx = end_idx
        end_idx   = end_idx + buffer_size

        packet_counter += len(packet_buffer)
        print('packets parsed: {}\tevents found: {}...'.format(packet_counter, event_counter),end='\r')


    if len(event_buffer) >= nhit_cut:
        evd_file.append(np.array(event_buffer))
        event_counter  += 1
    print('packets parsed: {}\tevents found: {}...Done!'.format(packet_counter, event_counter))

    print('finishing up...')
    evd_file.close()
    larpix_logfile.close()
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_filename','-i',required=True,type=str)
    parser.add_argument('--out_filename','-o',required=True,type=str)
    parser.add_argument('--geometry_file','-g',default=_default_geometry_file,type=str,help='''default=%(default)s''')
    parser.add_argument('--pedestal_file','-p',default=_default_pedestal_file,type=str,help='''default=%(default)s''')
    parser.add_argument('--configuration_file','-c',default=_default_configuration_file,type=str,help='''default=%(default)s''')
    parser.add_argument('--buffer_size','-b',default=_default_buffer_size,type=int,help='''default=%(default)s''')
    parser.add_argument('--event_dt',default=_default_event_dt,type=int,help='''default=%(default)s''')
    parser.add_argument('--nhit_cut',default=_default_nhit_cut,type=int,help='''default=%(default)s''')
    parser.add_argument('--max_packets','-n',default=_default_max_packets,type=int,help='''default=%(default)s''')
    parser.add_argument('--vd',default=_default_vd,type=float,help='''default=%(default)s''')
    parser.add_argument('--clock_period',default=_default_clock_period,type=float,help='''default=%(default)s''')
    parser.add_argument('--dbscan_eps',default=_default_dbscan_eps,type=float,help='''default=%(default)s''')
    parser.add_argument('--dbscan_min_samples',default=_default_dbscan_min_samples,type=int,help='''default=%(default)s''') 
    args = parser.parse_args()
    main(**vars(args))
