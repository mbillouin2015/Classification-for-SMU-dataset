%matplotlib  qt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

#df = pd.read_csv(path + 'utah_data.csv')
df = pd.read_csv('src/SMU_site.csv')
latmin = min(df['lat']) - 1
latmax = max(df['lat']) + 0.1
lomin = min(df['lon']) - 0.1
lomax = max(df['lon']) + 1

# PLOT SMU STATIONS AND EVENTS
fig,ax = plt.subplots()

rmt = df[df['type']=='RMT']
rmt_origin = df[df['type']=='RMT_SITE']

m = Basemap(llcrnrlat=latmin,urcrnrlat=latmax,llcrnrlon=lomin,
    urcrnrlon=lomax,resolution='h',epsg = 32142)
m.fillcontinents(color='tan',lake_color='k',zorder=0)
m.drawstates()
m.drawcountries()

parallels = np.arange(35,48,1)
m.drawparallels(parallels,linewidth=0.1,labels=[True,False,False,False])
meridians = np.arange(-120,-105,2)
m.drawmeridians(meridians,linewidth = 0.1,labels=[False,False,False,True])

for i in range(len(rmt)):
    x,y = m(rmt.iloc[i]['lon'],rmt.iloc[i]['lat'])
    plt.scatter(x,y,facecolor='cornflowerblue',edgecolor='k',s=35,
                zorder = 50000)
plt.scatter(x,y,facecolor='cornflowerblue', edgecolor='k', s=35, zorder=50000,
            label='Station Location')


#fig,ax = plt.subplots()
uttr = df[df['type']=='UTTR']
uttr_origin = df[df['type']=='UTTR_SITE']


for i in range(len(uttr)):
    x,y = m(uttr.iloc[i]['lon'],uttr.iloc[i]['lat'])
    plt.scatter(x,y,facecolor='cornflowerblue',edgecolor='k',s=35, zorder = 50000)

x,y = m(uttr_origin.iloc[0].lon,uttr_origin.iloc[0].lat)
plt.scatter(x,y,facecolor='y',edgecolor='k',s=250, zorder = 50000,
            label='RBD Origin', marker="*")

x,y = m(rmt_origin.iloc[0].lon,rmt_origin.iloc[0].lat)
plt.scatter(x,y,facecolor='red',edgecolor='k',s=250, zorder = 50000,
            label='RMT Origin', marker="*")

plt.legend(loc=1)
plt.savefig('figures/smu_site.svg')
plt.show()




