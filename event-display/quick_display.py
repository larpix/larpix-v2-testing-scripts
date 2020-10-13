import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py
import argparse
plt.ion()

def vol3d(x,y,z,q,*geom,name=None,fig=None):
    xyz = np.array(list(zip(x,y,z)))
    vox_q, _ = np.histogramdd(xyz, weights=q,
        bins=(
            np.linspace(geom[0],geom[1],
                int((geom[1]-geom[0])/geom[-2])+1),
            np.linspace(geom[2],geom[3],
                int((geom[3]-geom[2])/geom[-2])+1),
            np.linspace(geom[4],geom[5],
                int((geom[5]-geom[4])/geom[-1])+1),
            ))
    norm = lambda x: (x - min(np.min(x),0)) / (np.max(x) - np.min(x))
    cmap = plt.cm.get_cmap('plasma')
    vox_color = cmap(norm(vox_q))
    vox_color[..., 3] = norm(vox_q)

    ax = fig.add_subplot('122', projection='3d')
    ax.voxels(vox_q, facecolors=vox_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    plt.tight_layout()

    plt.show()
    return fig

def proj2d(x,y,q,*geom,name=None,fig=None):
    ax = fig.add_subplot('221')
    h = ax.hist2d(x,y,bins=(
        np.linspace(geom[0],geom[1],int((geom[1]-geom[0])/geom[-2])+1),
        np.linspace(geom[2],geom[3],int((geom[1]-geom[0])/geom[-2])+1)
        ),
        weights=q,
        cmin=0.0001,
        cmap='plasma'
    )
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.tight_layout()

    plt.show()
    plt.colorbar(h[3],label='charge [ke]')
    return fig

def proj_time(t,q,*geom,name=None,fig=None):
    ax = fig.add_subplot('223')
    ax.hist(t, weights=q,
        bins=np.linspace(geom[4],geom[5],
                int((geom[5]-geom[4])/geom[-1])+1),
        histtype='step', label='binned')
    plt.xlabel('timestamp [0.1us]')
    plt.ylabel('charge [ke]')
    plt.tight_layout()

    plt.show()
    return fig

def hit_times(t,q,*geom,name=None,fig=None):
    ax = fig.add_subplot('223')
    t,q = zip(*sorted(zip(t[t<geom[5]],q[t<geom[5]])))
    ax.plot(t,q,'r.', label='hits')
    plt.xlabel('timestamp [0.1us]')
    plt.ylabel('charge [ke]')
    plt.legend()
    plt.tight_layout()

    plt.show()
    return fig

def generate_plots(event, f, geom=[], fig=None):
    hits = f['hits']

    hit_ref = event['hit_ref']

    x = hits[hit_ref]['px']
    y = hits[hit_ref]['py']
    z = hits[hit_ref]['ts'] - event['ts_start']
    q = hits[hit_ref]['q'] * 0.250

    name = 'Event {}/{} ({})'.format(event['evid'],len(f['events']),f.filename)
    if not fig:
        fig = plt.figure(name)
    fig = vol3d(x,y,z,q,*geom,name=name,fig=fig)
    fig = proj2d(x,y,q,*geom,name=name,fig=fig)
    fig = proj_time(z,q,*geom,name=name,fig=fig)
    fig = hit_times(z,q,*geom,name=name,fig=fig)
    fig.canvas.set_window_title(name)
    return fig

def open_file(filename):
    return h5py.File(filename,'r')

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',required=True,help='''
    Input event display file
    ''')
parser.add_argument('--nhit_sel',default=0, type=int, help='''
    Optional, sub-select on nhit greater than this value
    ''')
parser.add_argument('--geom_limits', default=[-48.774,48.774,-48.774,48.774,0,1000,4.434,23], nargs=8, type=float, metavar=('XMIN','XMAX','YMIN','YMAX','TMIN','TMAX','PIXEL_PITCH','TIME_VOXEL'), help='''
    Optional, limits for geometry
    ''')
args = parser.parse_args()

f = open_file(args.input)
events = f['events']
hits = f['hits']
fig = None
ev = 0
while True:
    print('displaying event {} with nhit_sel={}'.format(ev,args.nhit_sel))
    if ev >= np.sum(events['nhit'] > args.nhit_sel):
        exit()
    event = events[events['nhit'] > args.nhit_sel][ev]
    fig = generate_plots(event, f, args.geom_limits, fig=fig)
    print('Event:',event)
    print('Hits:',hits[event['hit_ref']])
    user_input = input('Next event (q to exit/enter for next/number to skip to position)?\n')
    if user_input == '':
        ev += 1
    elif user_input[0].lower() == 'q':
        exit()
    else:
        ev = int(user_input)
    plt.clf()
