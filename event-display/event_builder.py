import numpy as np


class EventBuilder(object):
    '''
        Base class for event builder algorithms

        Initial config is provided by the `--event_builder_config {}` argument
        passed to the `to_evd_file.py` script

    '''
    def __init__(self, **params):
        '''
            Initialize given parameters for the class, each parameter is
            optional with a default provided by the implemented class
        '''
        pass

    def get_config(self):
        '''
            :returns: a `dict` of the instance configuration
        '''
        return dict()

    def build_events(self, packets, unix_ts):
        '''
            Run the event builder on a sub-set of packet-formatted array data
            The unix timestamp for each packet is provided as additional meta-data

            :returns: a `tuple` of `lists` of the packet array grouped into events, along with their corresponding unix timestamps
        '''
        raise NotImplementedError('Event building for this class has not been implemented!')


class TimeDeltaEventBuilder(EventBuilder):
    '''
        Original "gap-based" event building

        Searches for separations in data greater than the `event_dt` parameter.
        Events are formed at these boundaries. Any events that are greater than
        `max_event_dt` in length are broken up into separate events at the
        `max_event_dt` boundaries.

        Configurable parameters::

            event_dt        - gap size to separate into different events
            max_event_dt    - maximum event length

    '''
    default_event_dt = 1820
    default_max_event_dt = 1820 * 3

    def __init__(self, **params):
        super(self).__init__(**params)
        self.event_dt = params.get('event_dt', self.default_event_dt)
        self.max_event_dt = params.get('max_event_dt', self.default_max_event_dt)

        self.event_buffer = np.empty((0,)) # keep track of partial events from previous calls
        self.event_buffer_unix_ts = np.empty((0,))

    def get_config(self):
        return dict(
            event_dt=self.event_dt,
            max_event_dt=self.max_event_dt
            )

    def build_events(self, packets, unix_ts):
        # sort packets to fix 512 bug
        sorted_idcs = np.argsort(np.append(self.event_buffer, packets), order='timestamp')
        packets     = packets[sorted_idcs]
        unix_ts     = np.append(self.event_buffer_unix_ts, unix_ts)[sorted_idcs]

        # cluster into events by delta t
        packet_dt = packets['timestamp'][1:] - packets['timestamp'][:-1]
        event_idx     = np.argwhere(np.abs(packet_dt) > self.event_dt).flatten() - 1
        events        = np.split(packets, event_idx)
        event_unix_ts = np.split(unix_ts, event_idx)

        # reserve last event of every chunk for next iteration
        if len(events):
            self.event_buffer = np.copy(events[-1])
            self.event_buffer_unix_ts = np.copy(event_unix_ts[-1])
            del events[-1]
            del event_unix_ts[-1]

        # break up events longer than max window
        i = 0
        while i < len(events) and len(events[i]) \
                and events[i]['timestamp'][-1] - events[i]['timestamp'][0] > self.max_event_dt:
            event0, event1, unix_ts0, unix_ts1 = self.split_at_timestamp(
                events[i]['timestamp'][0] + self.max_event_dt,
                events[i],
                event_unix_ts[i]
                )
            events[i] = event0
            events.insert(i+1, event1)
            event_unix_ts[i] = unix_ts0
            event_unix_ts.insert(i+1, unix_ts1)
            i += 1

        return events, event_unix_ts

    @staticmethod
    def split_at_timestamp(timestamp,event,*args):
        '''
        Breaks event into two arrays at index where event['timestamp'] > timestamp
        Additional arrays can be specified with kwargs and will be split at the same
        index

        :returns: tuple of two event halves followed by any additional arrays (in pairs)
        '''
        args = list(args)
        timestamps = event['timestamp'].astype(int)
        indices = np.argwhere(timestamps > timestamp)
        if len(indices):
            idx = np.min(indices)
            args.insert(0,event)
            rv = [(arg[:idx], arg[idx:]) for arg in args]
            return tuple(v for vs in rv for v in vs)
        args.insert(0,event)
        rv = [(arg, np.array([], dtype=arg.dtype)) for arg in args]
        return tuple(v for vs in rv for v in vs)


class SymmetricWindowEventBuilder(EventBuilder):
    '''
        A sliding-window based event builder.

        Calculates the time difference between each packet in the chunk. Two
        packets are deemed "correlated" if they are separated by less than the
        `window`. Events are formed as contiguous regions in time where the
        number of correlated packets exceeds the `threshold`.

        Configurable parameters::

            window      - time delta to consider as correlated
            threshold   - number of correlated hits to initiate event

    '''
    default_window = 1820 * 2
    default_threshold = 25

    def __init__(self, **params):
        super(self).__init__(**params)
        self.window = params.get('window', self.default_window)
        self.threshold = params.get('threshold', self.default_threshold)

        self.event_buffer = np.empty((0,)) # keep track of partial events from previous calls
        self.event_buffer_unix_ts = np.empty((0,))

    def get_config(self):
        return dict(
            window=self.window,
            threshold=self.threshold
            )

    def build_events(self, packets, unix_ts):
        # sort packets to fix 512 bug
        sorted_idcs = np.argsort(np.append(self.event_buffer, packets), order='timestamp')
        packets     = packets[sorted_idcs]
        unix_ts     = np.append(self.event_buffer_unix_ts, unix_ts)[sorted_idcs]

        # calculate time distance between hits
        timestamp       = packets['timestamp'].astype(int)
        ts_diff_matrix  = np.abs(timestamp.reshape(-1,1) - timestamp.reshape(1,-1))

        # find high correlation regions
        n_corr_hits     = np.count_nonzero(ts_diff_matrix <= self.window, axis=-1)
        is_in_event     = (n_corr_hits >= self.threshold).astype(int)
        event_boundary  = np.diff(is_in_event)

        # find rising/falling edges
        event_idcs    = np.argwhere(event_boundary != 0).flatten() + 1

        if is_in_event[0]:
            # first packet is in first event, make sure you align the start/end idcs correctly
            event_idcs = np.r_[0, event_idcs]

        if is_in_event[-1]:
            # last event is incomplete, reserve for next iteration
            self.event_buffer = packets[event_idcs[-1]:]
            self.event_buffer_unix_ts = unix_ts[event_idcs[-1]:]
            packets = packets[:event_idcs[-1]]
            unix_ts = unix_ts[:event_idcs[-1]]
            event_idcs = event_idcs[:-1]

        # break up by event
        events = np.split(packets, event_idcs)
        event_unix_ts = np.split(unix_ts, event_idcs)
        is_event = np.r_[False, is_in_event[event_idcs]]

        # only return packets from events
        return zip(*[(event, unix_ts) for i, (event,unix_ts) in enumerate(zip(events, event_unix_ts)) if is_event[i]])


