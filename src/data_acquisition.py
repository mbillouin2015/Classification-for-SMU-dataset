import numpy as np
import pandas as pd
import os
from tqdm import trange, tqdm
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client

# client = Client('IRIS', timeout=1920)
# starttime = UTCDateTime('2007-05-15T19:30:42')
# endtime = starttime + 10
#
# bulk = [('*', '*', '*', '*', starttime, endtime)]
#
# st = client.get_stations_bulk(bulk)
#
df = pd.read_pickle('/home/mbillouin2015/PycharmProjects/Classification-for-SMU-dataset/data/smu_v4_staged.pickle')
