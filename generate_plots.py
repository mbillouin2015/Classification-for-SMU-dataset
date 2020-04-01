import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import sys
import pandas as pd
from obspy.core.trace import Trace

DATADIR = 'data/SMU_V5.pickle'
SAVEDIR = 'plots/'

# sys.stdout = open('output.txt', 'w')

print('INFO: Reading Dataframe.')
data = pd.read_pickle(DATADIR)
print('INFO: complete')
data.type.replace(to_replace='RBD', value='UTTR', inplace=True)
data.type.replace(to_replace='RMB', value='RMT', inplace=True)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5))

for idx in trange(len(data)):
    ax.clear()
    row = data.loc[idx]
    if row.type == 'RMT':
        color = 'r'
    else:
        color ='b'

    new_SAVEDIR = SAVEDIR + row.type + '/'
    tr = Trace()
    tr.data = row.ts
    tr.stats.starttime = row.datetime
    tr.stats.sampling_rate = row.hz
    endtime = tr.stats.endtime

    t = np.arange(0.0, len(row.ts)/row.hz, 1/row.hz)
    ax.plot(t, row.ts, color=color)
    ax.set_xlim(xmin=0, xmax=len(row.ts) / row.hz)
    ax.set(ylabel='Amplitude', xlabel='Time [s]')
    title = str(row.datetime) + ' - ' + str(endtime)
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    # plt.savefig(new_SAVEDIR + str(idx) +'.png')
    # plt.show()
plt.close()

@jit
def plotDF(df, save_dir):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5))
    for idx in trange(len(df)):
        row = df.loc[idx]
        ax.clear()
        ax.plot(row.ts)
        ax.set_xlim(xmin=0)
        ax.set(ylabel='Amplitude', xlabel='Samples')
        plt.savefig(save_dir +str(idx) + '.png')

    plt.close(fig)