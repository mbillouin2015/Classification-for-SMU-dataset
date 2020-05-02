import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import trange
import sys
import pandas as pd
from obspy.core.trace import Trace

# DATADIR = 'data/SMU.pickle'
DATADIR = 'data/FordA.pickle'
SAVEDIR = 'plots/FordA/'

# sys.stdout = open('output.txt', 'w')

print('INFO: Reading Dataframe.')
data = pd.read_pickle(DATADIR)
print('INFO: complete')
# data.type.replace(to_replace='RBD', value='UTTR', inplace=True)
# data.type.replace(to_replace='RMB', value='RMT', inplace=True)


@jit
def plotDF(data, SAVEDIR):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5))
    for idx in trange(len(data)):
        row = data.loc[idx]
        ax.clear()
        ax.plot(row.ts)
        ax.set_xlim(xmin=0)
        ax.set(ylabel='Amplitude', xlabel='Samples')
        plt.savefig(SAVEDIR + 'class-'+ str(row.class_lbl) + '-' + str(idx) + '.png')

    plt.close(fig)

plotDF(data=data, SAVEDIR=SAVEDIR)