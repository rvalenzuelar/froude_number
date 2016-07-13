

import numpy as np
import matplotlib.pyplot as plt
#import os
import Meteoframes as mf
import pandas as pd
import seaborn as sns

from matplotlib import rcParams
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['legend.handletextpad'] = 0.1
rcParams['mathtext.default'] = 'sf'

#home = os.path.expanduser('~')
home = '/localdata'

'''
 Case 09 is the only with data available for TTA
 period (balloon BVF and X-Pol obs)
'''


def sin(val):
    return np.sin(np.radians(val))

def cos(val):
    return np.cos(np.radians(val))

def sqrt(val):
    return np.sqrt(val)

def squeeze(val):
    return np.squeeze(val)

def get_bvf(flist=None,stat=None,layer=None):
    
    ftxt = home + '/BALLOON/all/{}.tsv'
    stats = []
    for name in flist:
        df = mf.parse_sounding2(ftxt.format(name))
        bvf = pd.DataFrame(index=df.index, columns=['bvf'])
        dry_cond = df['RH'] < 90
        bvf['bvf'][dry_cond] = df['bvf_dry'][dry_cond]
        bvf['bvf'][~dry_cond] = df['bvf_moist'][~dry_cond]
        if stat == 'median':
            stats.append(sqrt(bvf.loc[layer[0]:layer[1]].median()))
        elif stat == 'mean':        
            stats.append(sqrt(bvf.loc[layer[0]:layer[1]].mean()))
        elif stat == 'max':
            stats.append(sqrt(bvf.loc[layer[0]:layer[1]].max()))
    return squeeze(np.array(stats))

def main():
    minU, maxU = [0, 20]
    minN, maxN = [5e-5, 5e-2]
    U = np.linspace(minU, maxU, 1000)  # [m s-1]
    N = np.linspace(minN, maxN, 1000)  # [s-1]
    
    UU, NN = np.meshgrid(U, N)
    
    Fr0500 = UU/(NN*500)
    Fr1000 = UU/(NN*1000)
    
    ''' estimates of spd and dir from PPIs and RHIs at range
    where maximum Doppler is clear; usually it is at
    the same time than sounding but if not clear then
    choosed the neares PPI time. If Doppler is absent
    or not clear within an hour then uses NaN '''
    ppi_wspd_c09 = np.array([15, 15, 18, 15, 13]+[np.nan]*3)
    ppi_wdir_c09 = np.array([170, 170, 170, 180, 200]+[np.nan]*3)
    rhi_wspd_c09 = np.array([10, 10, 15, 10, 10]+[np.nan]*3)

    ppi_wspd_c13 = np.array([30, 25]+[np.nan]*7)
    ppi_wdir_c13 = np.array([170, 180]+[np.nan]*7)
    rhi_wspd_c13 = np.array([25, 25]+[np.nan]*7) #  c13 has only 2 rhis in 
                                                 #  180 azimuth
    
    ''' estimates components and upslope wind '''
    u_comp = -ppi_wspd_c09*sin(ppi_wdir_c09)
    v_comp = rhi_wspd_c09
    Up_c09 = -(u_comp*sin(230)+v_comp*cos(230))
    u_comp = -ppi_wspd_c13*sin(ppi_wdir_c13)
    v_comp = rhi_wspd_c13
    Up_c13 = -(u_comp*sin(230)+v_comp*cos(230))   
    
    ''' gets BVFs '''
    
    stat='median'
    sfiles = ['20030121_070405', '20030121_090335', '20030121_105906',
              '20030121_130221', '20030121_235258', '20030122_040333',
              '20030122_065918', '20030122_172227']
    medianc09 = np.empty((2,8))    
    medianc09[0,:] = get_bvf(flist=sfiles,stat=stat,layer=[0,500])
    medianc09[1,:] = get_bvf(flist=sfiles,stat=stat,layer=[0,1000])

    sfiles = ['20040216_1758','20040216_2043','20040217_1510',
              '20040217_1658','20040217_1859','20040217_2058',
              '20040217_2159','20040217_2259','20040217_2359',]
    medianc13 = np.empty((2,9))    
    medianc13[0,:] = get_bvf(flist=sfiles,stat=stat,layer=[0,500])
    medianc13[1,:] = get_bvf(flist=sfiles,stat=stat,layer=[0,1000])
    
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(6, 6))
    
    c = ax.contour(UU, NN, Fr0500, [1.0], colors='k',linestyles=':')
    plt.clabel(c, fmt='h = 0.5 km', manual=[(15, 0.03)])
    
    c = ax.contour(UU, NN, Fr1000, [1.0], colors='k',linestyles='-')
    plt.clabel(c, fmt='h = 1.0 km', manual=[(15, 0.03)]) 
    
    ax.scatter(Up_c09, medianc09[0], marker='+', s=100, color='b',lw=2,
               label='21-23Jan03 0.5 km'.format(stat))
    
    ax.scatter(Up_c13, medianc13[0], marker='x', s=100, color='b',lw=2,
                label='16-18Feb04 0.5 km'.format(stat))
    
    ax.scatter(Up_c09, medianc09[1], marker='+', s=100, color='g',lw=2,
               label='21-23Jan03 1.0 km'.format(stat))  
    
    ax.scatter(Up_c13, medianc13[1], marker='x', s=100, color='g',lw=2,
                label='16-18Feb04 1.0 km'.format(stat))

    ax.annotate("",
                xy=(0.64, 0.67), xycoords='figure fraction',
                xytext=(0.74, 0.62), textcoords='figure fraction',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=0.3",
                                lw=2),
                )
    ax.text(0.6,0.66,'blocked',transform=ax.transAxes,
            ha='right',fontsize=15)
    ax.annotate("",
                xy=(0.72, 0.46), xycoords='figure fraction',
                xytext=(0.75, 0.58), textcoords='figure fraction',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.3",
                                lw=2),
                )
    ax.text(0.72,0.41,'unblocked',transform=ax.transAxes,
            ha='center',va='top',fontsize=15)

    ax.set_xlim([minU, maxU])
    ax.set_ylim([minN, maxN])
    
    plt.legend(loc=2, scatterpoints=1,fontsize=15)
    plt.xlabel(r'U [$m s^{-1}$]')
    plt.ylabel(r'N [$s^{-1}$]')
    plt.title('Froude number (Fr=U/Nh)',fontsize=15)
    
#    plt.show()
    
    fname='/home/raul/Desktop/froude_number_{}.png'.format(stat)
    plt.savefig(fname, dpi=300, format='png',papertype='letter',
                bbox_inches='tight')


main()