import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import Meteoframes as mf
import pandas as pd

'''
 Case 09 is the only with data available for TTA
 period (balloon BVF and X-Pol obs)
'''


def sin(val):
    return np.sin(np.radians(val))


def cos(val):
    return np.cos(np.radians(val))

home = os.path.expanduser('~')

minU, maxU = [0, 20]
minN, maxN = [5e-5, 5e-2]
U = np.linspace(minU, maxU, 1000)  # [m s-1]
N = np.linspace(minN, maxN, 1000)  # [s-1]

UU, NN = np.meshgrid(U, N)

Fr0500 = UU/(NN*500)
Fr1000 = UU/(NN*1000)

''' estimates of spd and dir from PPIs at range
where maximum Doppler is clear; usually is at
the same time than sounding but if not clear then
choosed the neares PPI time. If Doppler is absent
or not clear within an hour then uses NaN '''
ppi_wspd = np.array([15, 15, 18, 15, 13, np.nan, np.nan, np.nan])
ppi_wdir = np.array([170, 170, 170, 180, 200, np.nan, np.nan, np.nan])
u_comp = -ppi_wspd*sin(ppi_wdir)

''' estimates of spd from RHIs '''
rhi_wspd = np.array([10, 10, 15, 10, 10, np.nan, np.nan, np.nan])
v_comp = rhi_wspd

''' estimate upslope wind '''
Up = -(u_comp*sin(230)+v_comp*cos(230))

''' gets BVFs '''
sfiles = ['21_070405', '21_090335', '21_105906',
          '21_130221', '21_235258', '22_040333',
          '22_065918', '22_172227']
ftxt = home + '/BALLOON/case09/FRS_200301{}.tsv'

median500 = []
mean500 = []
max500 = []
median1000 = []
mean1000 = []
max1000 = []
for name in sfiles:
    df = mf.parse_sounding2(ftxt.format(name))
    bvf = pd.DataFrame(index=df.index, columns=['bvf'])
    dry_cond = df['RH'] < 90
    bvf['bvf'][dry_cond] = df['bvf_dry'][dry_cond]
    bvf['bvf'][~dry_cond] = df['bvf_moist'][~dry_cond]

    median500.append(np.sqrt(bvf.loc[0:500].median()))
    mean500.append(np.sqrt(bvf.loc[0:500].mean()))
    max500.append(np.sqrt(bvf.loc[0:500].max()))
    median1000.append(np.sqrt(bvf.loc[0:1000].median()))
    mean1000.append(np.sqrt(bvf.loc[0:1000].mean()))
    max1000.append(np.sqrt(bvf.loc[0:1000].max()))

median500 = np.squeeze(np.array(median500))
mean500 = np.squeeze(np.array(mean500))
max500 = np.squeeze(np.array(max500))
median1000 = np.squeeze(np.array(median1000))
mean1000 = np.squeeze(np.array(mean1000))
max1000 = np.squeeze(np.array(max1000))

fig, ax = plt.subplots(figsize=(8, 7))
c = ax.contour(UU, NN, Fr0500, [1.0], colors='b')
plt.clabel(c, fmt='H = 0.5 km', manual=[(15, 0.03)])
c = ax.contour(UU, NN, Fr1000, [1.0], colors='g')
plt.clabel(c, fmt='H =1.0 km', manual=[(15, 0.03)])
ax.scatter(Up, mean500, marker='o', s=100, color='b',
           facecolor='none', label='mean 0.5 km')
ax.scatter(Up, mean1000, marker='o', s=100, color='g',
           facecolor='none', label='mean 1.0 km')
ax.scatter(Up, median500, marker='+', s=100, color='b',
           label='median 0.5 km')
ax.scatter(Up, median1000, marker='+', s=100, color='g',
           label='median 1.0 km')
# ax.scatter(Up, max500, marker='x', s=100, color='b',
#            label='max 0.5 km')
# ax.scatter(Up, max1000, marker='x', s=100, color='g',
#            label='max 1.0 km')
ax.set_xlim([minU, maxU])
ax.set_ylim([minN, maxN])

plt.legend(loc=2, scatterpoints=1)
# plt.axhline(0.007, color='r', ls='--')
# plt.axhline(0.025, color='r', ls='--')
plt.xlabel('U [m s-1]')
plt.ylabel('N [s-1]')
plt.title('Storm 2 Froude number (Fr=U/NH)')
plt.show(block=False)
