import h5py
import threading
import os
import numpy as np
from collections import defaultdict
import queue
import json

region_ref = h5py.special_dtype(ref=h5py.RegionReference)

class LArPixEVDFile(object):
    dtype_desc = {
        'info' : None,
        'hits' : [
            ('hid', 'i8'),
            ('px', 'i8'), ('py', 'i8'), ('ts', 'i8'), ('q', 'i8'),
            ('iochain', 'i8'), ('chipid', 'i8'), ('channelid', 'i8'),
            ('geom', 'i8'), ('event_ref', region_ref), ('track_ref', region_ref)],
        'events' : [
            ('evid', 'i8'), ('track_ref', region_ref), ('hit_ref', region_ref),
            ('nhit', 'i8'), ('q', 'i8'), ('ts_start', 'i8'), ('ts_end', 'i8')],
        'tracks' : [
            ('track_id','i8'), ('event_ref', region_ref), ('hit_ref', region_ref),
            ('theta', 'f8'),
            ('phi', 'f8'), ('xp', 'f8'), ('yp', 'f8'), ('nhit', 'i8'),
            ('q', 'i8'), ('ts_start', 'i8'), ('ts_end', 'i8'),
            ('sigma_theta', 'f8'), ('sigma_phi', 'f8'), ('sigma_x', 'f8'),
            ('sigma_y', 'f8'), ('length', 'f8'), ('start', '(3,)f8'),
            ('end', '(3,)f8')],
    }

    def __init__(self, filename, configuration_file=None, geometry_file=None, pedestal_file=None, buffer_len=1024):
        self.is_open = True
        if os.path.exists(filename):
            raise OSError('{} exists!'.format(filename))
        self.h5_file = h5py.File(filename)
        self.out_buffer = list()
        self.buffer_len = buffer_len
        self.geometry = defaultdict(lambda: (0.,0.))
        if geometry_file is not None:
            import larpixgeometry.layouts
            geo = larpixgeometry.layouts.load(geometry_file) # open geometry yaml file
            for chip, pixels in geo['chips']:
                for channel, pixel_id in enumerate(pixels):
                    if pixel_id is not None:
                        self.geometry[(chip,channel)] = geo['pixels'][pixel_id][1:3]
        self.pedestal = defaultdict(lambda: dict(
            pedestal_mv=550
            ))
        if pedestal_file is not None:
            with open(pedestal_file,'r') as infile:
                for key,value in json.load(infile).items():
                    self.pedestal[key] = value
        self.configuration = defaultdict(lambda: dict(
            vref_mv=1500,
            vcm_mv=550
            ))
        if configuration_file is not None:
            with open(configuration_file,'r') as infile:
                for key,value in json.load(infile).items():
                    self.configuration[key] = value

        self._queue  = queue.Queue()
        self._outfile_worker = threading.Thread(target=self._parse_events_array)

        self._create_datasets()

    def append(self, event_array):
        self.out_buffer.append(event_array)
        if len(self.out_buffer) >= self.buffer_len:
            self.flush()

    def flush(self, block=False):
        if not len(self.out_buffer):
            return
        self._queue.put(self.out_buffer)
        if not self._outfile_worker.isAlive():
            self._outfile_worker = threading.Thread(target=self._parse_events_array)
            self._outfile_worker.start()
        if block:
            self._outfile_worker.join()
        self.out_buffer = list()

    def close(self):
        if self.is_open:
            self.flush(block=True)
            self.is_open = False
            self.h5_file.close()

    def _create_datasets(self):
        for dataset_name, dataset_dtype in self.dtype_desc.items():
            if not dataset_dtype is None:
                self.h5_file.create_dataset(dataset_name, (0,), maxshape=(None,), dtype=dataset_dtype)
            else:
                self.h5_file.create_group(dataset_name)

    def _get_pixel_id(self, chip_id, channel_id):
        for chip_info in self.geometry['chips']:
            if chip_info[0] == chip_id:
                return chip_info[1][channel_id]
        return -1

    def _parse_events_array(self):
        while not self._queue.empty():
            events_list = list()
            try:
                events_list = self._queue.get(block=False)
            except queue.Empty:
                break
            events_dset = self.h5_file['events']
            tracks_dset = self.h5_file['tracks']
            hits_dset = self.h5_file['hits']
            events_idx = len(events_dset)
            tracks_idx = len(tracks_dset)
            hits_idx = len(hits_dset)
            events_dset.resize(events_dset.shape[0] + len(events_list), axis=0)
            tracks_dset.resize(tracks_dset.shape[0] + len(events_list), axis=0)
            hits_dset.resize(hits_dset.shape[0] + np.sum([len(event) for event in events_list]), axis=0)

            for event in events_list:
                events_dict = defaultdict(int)
                tracks_dict = defaultdict(int)
                hits_dict   = defaultdict(int)
                # create event info
                events_dict['nhit']     = len(event)
                events_dict['evid']     = events_idx
                events_dict['q']        = 0.
                events_dict['ts_start'] = event[0]['timestamp']
                events_dict['ts_end']   = event[-1]['timestamp']
                # create track info
                tracks_dict['track_id']    = tracks_idx
                tracks_dict['theta']       = 0
                tracks_dict['phi']         = 0
                tracks_dict['xp']          = 0
                tracks_dict['yp']          = 0
                tracks_dict['nhit']        = len(event)
                tracks_dict['q']           = 0
                tracks_dict['ts_start']    = event[0]['timestamp']
                tracks_dict['ts_end']      = event[-1]['timestamp']
                tracks_dict['sigma_theta'] = 0
                tracks_dict['sigma_phi']   = 0
                tracks_dict['sigma_x']     = 0
                tracks_dict['length']      = 0
                tracks_dict['start']       = (0.,0.,0.)
                tracks_dict['end']         = (0.,0.,0.)
                # create hit info
                hits_dict['hid']       = hits_idx + np.arange(len(event)).astype(int)
                hits_dict['px']        = np.zeros(len(event))
                hits_dict['py']        = np.zeros(len(event))
                if len(self.geometry.keys()):
                    xy = np.array([self.geometry[key] for key in zip(event['chip_id'], event['channel_id'])])
                    hits_dict['px']    = xy[:,0]
                    hits_dict['py']    = xy[:,1]
                hits_dict['ts']        = event['timestamp']
                hits_dict['q']         = np.zeros(len(event))
                hits_dict['iochain']   = event['io_channel']
                hits_dict['chipid']    = event['chip_id']
                hits_dict['channelid'] = event['channel_id']
                hits_dict['geom']      = np.zeros(len(event))
                hit_uniqueid = (event['io_group'].astype(int)-1)*256*256*64 \
                    + (event['io_channel'].astype(int)-1)*256*64 \
                    + (event['chip_id'].astype(int)*64) \
                    + (event['chip_id'].astype(int)*64)
                vref = np.array([self.configuration[str(unique_id)]['vref_mv'] for unique_id in hit_uniqueid])
                vcm = np.array([self.configuration[str(unique_id)]['vcm_mv'] for unique_id in hit_uniqueid])
                ped = np.array([self.pedestal[str(unique_id)]['pedestal_mv'] for unique_id in hit_uniqueid])
                q = event['dataword']/256*(vref-vcm) + vcm - ped
                hits_dict['q'] = q

                # calculate hit level info for tracks / events
                q_sum = np.sum(q)
                events_dict['q'] = q_sum
                tracks_dict['q'] = q_sum

                # create references
                event_ref = events_dset.regionref[events_idx]
                track_ref = tracks_dset.regionref[tracks_idx]
                hit_ref = hits_dset.regionref[hits_idx:hits_idx+len(event)]

                events_dict['track_ref'] = track_ref
                events_dict['hit_ref'] = hit_ref
                tracks_dict['event_ref'] = event_ref
                tracks_dict['hit_ref'] = hit_ref
                hits_dict['event_ref'] = [event_ref]*len(event)
                hits_dict['track_ref'] = [track_ref]*len(event)

                self._fill('events', events_idx, 1, **events_dict)
                self._fill('tracks', events_idx, 1, **tracks_dict)
                self._fill('hits', hits_idx, len(event), **hits_dict)

                events_idx += 1
                tracks_idx += 1
                hits_idx   += len(event)

    def _fill(self, dset_name, start_idx, n, **kwargs):
        data = np.zeros(n, dtype=self.dtype_desc[dset_name])
        for key,val in kwargs.items():
            data[key] = val
        dset = self.h5_file[dset_name]
        dset[start_idx:start_idx+n] = data

def load_larpix_logfile(filename):
    datafile = h5py.File(filename, 'r')
    return datafile
