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
        super(TimeDeltaEventBuilder, self).__init__(**params)
        self.event_dt = params.get('event_dt', self.default_event_dt)
        self.max_event_dt = params.get('max_event_dt', self.default_max_event_dt)

        self.event_buffer = np.empty((0,)) # keep track of partial events from previous calls
        self.event_buffer_unix_ts = np.empty((0,), dtype='u8')

    def get_config(self):
        return dict(
            event_dt=self.event_dt,
            max_event_dt=self.max_event_dt
            )

    def build_events(self, packets, unix_ts):
        # sort packets to fix 512 bug
        packets     = np.append(self.event_buffer, packets) if len(self.event_buffer) else packets
        sorted_idcs = np.argsort(packets, order='timestamp')
        packets     = packets[sorted_idcs]
        unix_ts     = np.append(self.event_buffer_unix_ts, unix_ts)[sorted_idcs] if len(self.event_buffer_unix_ts) else unix_ts[sorted_idcs]

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

        Histograms the packets into bins of `window` width. Events are formed
        if a bin content is greater than `threshold`. The event extent covers
        the bin of interest and +/- 1 bin. If multiple adjacent bins exceed
        the threshold, they are merged into a single event.

        Configurable parameters::

            window      - bin width
            threshold   - number of correlated hits to initiate event

    '''
    default_window = 1820//2
    default_threshold = 10

    def __init__(self, **params):
        super(SymmetricWindowEventBuilder, self).__init__(**params)
        self.window = params.get('window', self.default_window)
        self.threshold = params.get('threshold', self.default_threshold)

        self.event_buffer = np.empty((0,)) # keep track of partial events from previous calls
        self.event_buffer_unix_ts = np.empty((0,), dtype='u8')

    def get_config(self):
        return dict(
            window=self.window,
            threshold=self.threshold
            )

    def build_events(self, packets, unix_ts):
        # sort packets to fix 512 bug
        packets     = np.append(self.event_buffer, packets) if len(self.event_buffer) else packets
        sorted_idcs = np.argsort(packets, order='timestamp')
        packets     = packets[sorted_idcs]
        unix_ts     = np.append(self.event_buffer_unix_ts, unix_ts)[sorted_idcs] if len(self.event_buffer_unix_ts) else unix_ts[sorted_idcs]

        # calculate time distance between hits
        min_ts, max_ts = np.min(packets['timestamp']), np.max(packets['timestamp'])
        bin_edges = np.linspace(min_ts, max_ts, int((max_ts - min_ts)//self.window))
        hist, bin_edges = np.histogram(packets['timestamp'], bins=bin_edges)

        # find high correlation regions
        event_mask = (hist > self.threshold)
        event_mask[:-1] = event_mask[:-1] | (hist > self.threshold)[1:]
        event_mask[1:] = event_mask[1:] | (hist > self.threshold)[:-1]

        # find rising/falling edges
        event_edges = np.diff(event_mask.astype(int))
        event_start_timestamp = bin_edges[1:-1][event_edges > 0]
        event_end_timestamp = bin_edges[1:-1][event_edges < 0]

        if not np.any(event_mask):
            # no events
            return [], []
        if not len(event_start_timestamp):
            # first packet starts event
            event_start_timestamp = np.r_[min_ts, event_start_timestamp]
        if not len(event_end_timestamp):
            # last packet ends event, keep for next but return no events
            mask = packets['timestamp'] > event_start_timestamp[-1]
            self.event_buffer = packets[mask]
            self.event_buffer_unix_ts = unix_ts[mask]
            packets = packets[~mask]
            unix_ts = unix_ts[~mask]
            event_start_timestamp = event_start_timestamp[:-1]
            return [], []

        if event_end_timestamp[0] < event_start_timestamp[0]:
            # first packet is in first event, make sure you align the start/end idcs correctly
            event_start_timestamp = np.r_[min_ts, event_start_timestamp]
        if event_end_timestamp[-1] < event_start_timestamp[-1]:
            # last event is incomplete, reserve for next iteration
            mask = packets['timestamp'] > event_start_timestamp[-1]
            self.event_buffer = packets[mask]
            self.event_buffer_unix_ts = unix_ts[mask]
            packets = packets[~mask]
            unix_ts = unix_ts[~mask]
            event_start_timestamp = event_start_timestamp[:-1]
        else:
            self.event_buffer = np.empty((0,), dtype=packets.dtype)
            self.event_buffer_unix_ts = np.empty((0,), dtype=unix_ts.dtype)

        # break up by event
        event_mask = (packets['timestamp'].reshape(1,-1) > event_start_timestamp.reshape(-1,1)) \
            & (packets['timestamp'].reshape(1,-1) < event_end_timestamp.reshape(-1,1))
        event_mask = np.any(event_mask, axis=0)
        event_diff = np.diff(event_mask, axis=-1)
        event_idcs = np.argwhere(event_diff).flatten() + 1

        events = np.split(packets, event_idcs)
        event_unix_ts = np.split(unix_ts, event_idcs)
        is_event = np.r_[False, event_mask[event_idcs]]

        # only return packets from events
        return zip(*[v for i,v in enumerate(zip(events, event_unix_ts)) if is_event[i]])


