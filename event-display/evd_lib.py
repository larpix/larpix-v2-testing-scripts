import h5py
import threading
import os
import numpy as np
import sklearn.cluster as cluster
import sklearn.decomposition as dcomp
from collections import defaultdict
import queue
import json

region_ref = h5py.special_dtype(ref=h5py.RegionReference)

class TrackFitter(object):
    def __init__(self, dbscan_eps=14, dbscan_min_samples=5, vd=1.648, clock_period=0.1):
        self._vd = vd
        self._clock_period = clock_period
        self._dbscan_eps = dbscan_eps
        self._dbscan_min_samples = dbscan_min_samples
        
        self._set_parameters(
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            vd=vd,
            clock_period=clock_period
            )
        self.pca = dcomp.PCA(n_components=1)

    def get_parameters(self, *args):
        '''
        Return ``dict`` of specified named parameters and values, if none specified, 
        return all parameters

        '''
        rv = dict()
        for key in ('vd','clock_period','z_scale','dbscan_eps','dbscan_min_samples'):
            if key in args or not args:
                rv[key] = getattr(self,'_{}'.format(key))
        return rv
        
    def _set_parameters(self, **kwargs):
        '''
        Sets parameters used in fitting tracks::
        
            vd: drift velocity [mm/us]
            clock_period: clock period for timestamp [us]
            dbscan_eps: epsilon used for clustering [mm]
            dbscan_min_samples: min samples used for clustering

        '''
        self._vd = kwargs.get('vd',self._vd)
        self._clock_period = kwargs.get('clock_period',self._clock_period)
        self._z_scale = self._vd * self._clock_period
        
        self._dbscan_eps = kwargs.get('dbscan_eps',self._dbscan_eps)
        self._dbscan_min_samples = kwargs.get('dbscan_min_samples',self._dbscan_min_samples)
        self.dbscan = cluster.DBSCAN(eps=self._dbscan_eps, min_samples=self._dbscan_min_samples)

    def fit(self, events, geometry, plot=False):
        event_tracks = list()
        for event in events:
            event_tracks.append(list())
            t0 = int(event['timestamp'][0])
            if not len(geometry.keys()): continue
            if len(event) < 2: continue

            # dbscan to find clusters
            xyz = np.array([(*geometry[(chip_id, channel_id)],(ts-t0)*self._z_scale) for chip_id, channel_id,ts in zip(event['chip_id'], event['channel_id'], event['timestamp'].astype(int))])
            clustering = self.dbscan.fit(xyz)
            track_ids = clustering.labels_

            if plot:
                print('plotting for dbscan tuning...')
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                def refit(self, eps):
                    self.dbscan.set_params(eps=eps)
                    labels = self.dbscan.fit(xyz).labels_
                    return len(np.unique(labels[labels!=-1]))
                eps = np.linspace(0.1,50,100)
                n = [refit(self,e) for e in eps]
                plt.ion()
                fig = plt.figure()
                fig.add_subplot(1,2,1)
                plt.plot(eps,n)
                ax = fig.add_subplot(1,2,2, projection='3d')
                ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2])
                plt.show()
                plt.pause(5)

            for track_id in np.unique(track_ids):
                if track_id == -1: continue
                # PCA on cluster hits
                mask = track_ids == track_id
                if np.sum(mask) < 2: continue
                centroid = np.mean(xyz[mask], axis=0)
                pca = self.pca.fit(xyz[mask] - centroid)
                axis = pca.components_[0] / np.linalg.norm(pca.components_[0])
                r_min,r_max = self._projected_limits(centroid, axis, xyz[mask])
                residual = self._track_residual(centroid, axis, xyz[mask])

                # convert back to clock ticks
                r_min = np.append(r_min,r_min[-1]/self._z_scale)
                r_max = np.append(r_max,r_max[-1]/self._z_scale)

                event_tracks[-1].append(dict(
                        track_id=track_id,
                        mask=mask,
                        centroid=centroid,
                        axis=axis,
                        residual=residual,
                        length=np.linalg.norm(r_max[:3]-r_min[:3]),
                        start=r_min,
                        end=r_max
                    ))
                # print(event_tracks[-1])
        return event_tracks

    def _projected_limits(self, centroid, axis, xyz):
        s = (xyz - centroid) @ axis
        r_max = centroid + axis * np.max(s)
        r_min = centroid + axis * np.min(s)
        return r_min,r_max

    def _track_residual(self, centroid, axis, xyz):
        s = (xyz - centroid) @ axis
        res = np.linalg.norm(xyz - (centroid + np.outer(s,axis)))
        return np.mean(res)

    def theta(self, axis):
        return np.arctan2(np.linalg.norm(axis[:2]),axis[-1])

    def phi(self, axis):
        return np.arctan2(axis[1],axis[0])

    def xyp(self, axis, centroid):
        if axis[-1] == 0: return centroid[:2]
        s = -centroid[-1] / axis[-1]
        return (centroid + axis * s)[:2]

class LArPixEVDFile(object):
    dtype_desc = {
        'info' : None,
        'hits' : [
            ('hid', 'i8'),
            ('px', 'f8'), ('py', 'f8'), ('ts', 'i8'), ('q', 'f8'),
            ('iochain', 'i8'), ('chipid', 'i8'), ('channelid', 'i8'),
            ('geom', 'i8'), ('event_ref', region_ref), ('track_ref', region_ref)],
        'events' : [
            ('evid', 'i8'), ('track_ref', region_ref), ('hit_ref', region_ref),
            ('nhit', 'i8'), ('q', 'f8'), ('ts_start', 'i8'), ('ts_end', 'i8'), ('ntracks', 'i8')],
        'tracks' : [
            ('track_id','i8'), ('event_ref', region_ref), ('hit_ref', region_ref),
            ('theta', 'f8'),
            ('phi', 'f8'), ('xp', 'f8'), ('yp', 'f8'), ('nhit', 'i8'),
            ('q', 'i8'), ('ts_start', 'i8'), ('ts_end', 'i8'),
            ('residual', 'f8'), ('length', 'f8'), ('start', 'f8', (4,)),
            ('end', 'f8', (4,))],
    }

    def __init__(self, filename, source_file=None, configuration_file=None, geometry_file=None,
                 pedestal_file=None, builder_config=None, fitter_config=None, buffer_len=128):
        self.is_open = True
        if os.path.exists(filename):
            raise OSError('{} exists!'.format(filename))
        self.h5_file = h5py.File(filename,'w')
        self.out_buffer = list()
        self.buffer_len = buffer_len
        
        self.geometry = defaultdict(lambda: (0.,0.))
        self.geometry_file = geometry_file
        if geometry_file is not None:
            import larpixgeometry.layouts
            geo = larpixgeometry.layouts.load(self.geometry_file) # open geometry yaml file
            for chip, pixels in geo['chips']:
                for channel, pixel_id in enumerate(pixels):
                    if pixel_id is not None:
                        self.geometry[(chip,channel)] = geo['pixels'][pixel_id][1:3]

        self.pedestal = defaultdict(lambda: dict(
            pedestal_mv=550
            ))
        self.pedestal_file = pedestal_file
        if pedestal_file is not None:
            with open(self.pedestal_file,'r') as infile:
                for key,value in json.load(infile).items():
                    self.pedestal[key] = value

        self.configuration = defaultdict(lambda: dict(
            vref_mv=1500,
            vcm_mv=550
            ))
        
        self.configuration_file = configuration_file
        if configuration_file is not None:
            with open(self.configuration_file,'r') as infile:
                for key,value in json.load(infile).items():
                    self.configuration[key] = value

        self.source_file = source_file

        fitter_config = fitter_config if fitter_config else dict()
        self.track_fitter = TrackFitter(**fitter_config)

        self._queue  = queue.Queue()
        self._outfile_worker = threading.Thread(target=self._parse_events_array)

        self._create_datasets()
        self._write_metadata(dict(
            info=dict(
                source_file=self.source_file,
                configuration_file=self.configuration_file,
                pedestal_file=self.pedestal_file,
                geometry_file=self.geometry_file
            ),
            hits=dict(),
            events=builder_config if builder_config else dict(),
            tracks=self.track_fitter.get_parameters()
        ))

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

    def _write_metadata(self, metadata):
        for name in metadata:
            for attr,value in metadata[name].items():
                self.h5_file[name].attrs[attr] = value

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
            event_tracks = self.track_fitter.fit(events_list, self.geometry)
            tracks_dset.resize(tracks_dset.shape[0] + np.sum([len(tracks) for tracks in event_tracks]), axis=0)
            hits_dset.resize(hits_dset.shape[0] + np.sum([len(event) for event in events_list]), axis=0)

            for event,tracks in zip(events_list,event_tracks):
                events_dict = defaultdict(int)
                tracks_dict = defaultdict(int)
                hits_dict   = defaultdict(int)
                # create event info
                events_dict['nhit']     = len(event)
                events_dict['ntracks']  = len(tracks)
                events_dict['evid']     = events_idx
                events_dict['q']        = 0.
                events_dict['ts_start'] = event[0]['timestamp'].astype(int)
                events_dict['ts_end']   = event[-1]['timestamp'].astype(int)
                # create hit info
                hits_dict['hid']       = hits_idx + np.arange(len(event)).astype(int)
                hits_dict['px']        = np.zeros(len(event))
                hits_dict['py']        = np.zeros(len(event))
                if len(self.geometry.keys()):
                    xy = np.array([self.geometry[key] for key in zip(event['chip_id'], event['channel_id'])])
                    hits_dict['px']    = xy[:,0]
                    hits_dict['py']    = xy[:,1]
                hits_dict['ts']        = event['timestamp'].astype(int)
                hits_dict['q']         = np.zeros(len(event))
                hits_dict['iochain']   = event['io_channel']
                hits_dict['chipid']    = event['chip_id']
                hits_dict['channelid'] = event['channel_id']
                hits_dict['geom']      = np.zeros(len(event))
                hit_uniqueid = (((event['io_group'].astype(int))*256 \
                            + event['io_channel'].astype(int))*256 \
                        + event['chip_id'].astype(int))*64 \
                    + event['channel_id'].astype(int)
                vref = np.array([self.configuration[str(unique_id)]['vref_mv'] for unique_id in hit_uniqueid])
                vcm = np.array([self.configuration[str(unique_id)]['vcm_mv'] for unique_id in hit_uniqueid])
                ped = np.array([self.pedestal[str(unique_id)]['pedestal_mv'] for unique_id in hit_uniqueid])
                q = event['dataword']/256. * (vref-vcm) + vcm - ped
                hits_dict['q'] = q.astype(int)
                # create track info
                if len(tracks):
                    tracks_dict['track_id']    = tracks_idx + np.arange(len(tracks)).astype(int)
                    tracks_dict['nhit']        = np.zeros((len(tracks),))
                    tracks_dict['q']           = np.zeros((len(tracks),))
                    tracks_dict['ts_start']    = np.zeros((len(tracks),))
                    tracks_dict['ts_end']      = np.zeros((len(tracks),))
                    tracks_dict['theta']       = np.zeros((len(tracks),))
                    tracks_dict['phi']         = np.zeros((len(tracks),))
                    tracks_dict['xp']          = np.zeros((len(tracks),))
                    tracks_dict['yp']          = np.zeros((len(tracks),))
                    tracks_dict['residual']    = np.zeros((len(tracks),))
                    tracks_dict['length']      = np.zeros((len(tracks),))
                    tracks_dict['start']       = np.zeros((len(tracks),4))
                    tracks_dict['end']         = np.zeros((len(tracks),4))
                    for i,track in enumerate(tracks):
                        tracks_dict['nhit'][i]      = np.sum(track['mask'])
                        tracks_dict['q'][i]         = np.sum(hits_dict['q'][track['mask']])
                        tracks_dict['theta'][i]     = self.track_fitter.theta(track['axis'])
                        tracks_dict['phi'][i]       = self.track_fitter.phi(track['axis'])
                        xp,yp = self.track_fitter.xyp(track['axis'],track['centroid'])
                        tracks_dict['xp'][i]        = xp
                        tracks_dict['yp'][i]        = yp
                        tracks_dict['start'][i]     = track['start']
                        tracks_dict['end'][i]       = track['end']
                        tracks_dict['residual'][i]  = track['residual']
                        tracks_dict['length'][i]    = track['length']
                        tracks_dict['ts_start'][i]  = hits_dict['ts'][track['mask']][0]
                        tracks_dict['ts_end'][i]    = hits_dict['ts'][track['mask']][-1]

                # calculate hit level info for events
                q_sum = np.sum(q)
                events_dict['q'] = q_sum

                # create references
                event_ref = events_dset.regionref[events_idx]
                track_ref = tracks_dset.regionref[tracks_idx:tracks_idx+len(tracks)]
                hit_ref = hits_dset.regionref[hits_idx:hits_idx+len(event)]

                events_dict['track_ref'] = track_ref
                events_dict['hit_ref'] = hit_ref
                tracks_dict['event_ref'] = [event_ref]*len(tracks)
                tracks_dict['hit_ref'] = [hits_dset.regionref[hits_dict['hid'][track['mask']]] for track in tracks]
                hits_dict['event_ref'] = [event_ref]*len(event)
                hits_dict['track_ref'] = [track_ref]*len(event)

                self._fill('events', events_idx, 1, **events_dict)
                self._fill('tracks', tracks_idx, len(tracks), **tracks_dict)
                self._fill('hits', hits_idx, len(event), **hits_dict)

                events_idx += 1
                tracks_idx += len(tracks)
                hits_idx   += len(event)

    def _fill(self, dset_name, start_idx, n, **kwargs):
        if n == 0: return
        data = np.zeros(n, dtype=self.dtype_desc[dset_name])
        for key,val in kwargs.items():
            data[key] = val
        dset = self.h5_file[dset_name]
        dset[start_idx:start_idx+n] = data

def load_larpix_logfile(filename):
    datafile = h5py.File(filename, 'r')
    return datafile
