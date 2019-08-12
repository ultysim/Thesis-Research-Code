import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib import pyplot as plt
import Analysis
import regression_cv as rcv
import FitRoutine2 as fr
import itertools
import os

dirpath = os.getcwd()

def plotFig(data,fitbit):
    xa = fitbit
    freq = xa[0]
    amp = xa[1] 
    phasel = xa[2]
    phaser = xa[3]
    height = xa[4]
    slope = xa[5]
    posleft = min(data[0])
    posright = max(data[0]) 
    v0 = xa[11]     
    xpos = np.linspace(posleft,posright,10000)
    ytargl = height + amp*np.sin(freq*xpos + phasel) + slope*(xpos - v0)
    ytargr = height + amp*np.sin(freq*xpos + phaser) + slope*(xpos - v0)
    plt.plot(data[0],data[1])
    plt.plot(xpos,ytargl,'k',linewidth=2)
    plt.plot(xpos,ytargr,'r',linewidth=2) 

def t(y_min,y_max):
    diff = y_max-y_min
    p = float(diff)*0.35
    return y_min-p,y_max+p

def init_plot(i):
    i.plot()
    plt.xlim([min(i.x),max(i.x)])
    plt.ylim([min(i.y)-0.05,max(i.y)+0.05])
    
def stitch(i,x):
    i.fits[1].plot_select('b',['k','r'],[2,2],i.fits[1].pos_left,x)
    i.fits[0].plot_select('b',['r','#00ff00'],[2,2],x,i.fits[0].pos_right)
    
def set_axis(sweep,x_min,x_max):
    plt.xlim(x_min,x_max)
    x = np.linspace(x_min,x_max)
    yl,yr = sweep.fits[j].yValue(x)
    y_min = np.min([yl,yr])
    y_max = np.max([yl,yr])
    l,h = t(y_min,y_max)
    plt.ylim(l,h)
    plt.yticks([y_min,y_max],(round(y_min,1),round(y_max,1)))
    x_low, x_high = np.percentile(x,[25,75])
    plt.xticks([int(x_low),int(x_high)])
    plt.tick_params(axis='y',pad=ypad)
    plt.tick_params(axis='x',pad=xpad)

def q_0():
    n = 2
    global j
    j = 2
    init_plot(i14[n])
    i14[n].fits[2].plot_select('l','k',lw,-21.4367,-19.6676)
    i14[n].fits[2].plot_select('l','k',lw,-19.6676,-13.76,'--')

    i14[n].fits[2].plot_select('r','r',lw,-21.4367,-19.6676,'--')
    i14[n].fits[2].plot_select('r','r',lw,-19.6676,-18.6335)
    i14[n].fits[2].plot_select('r','r',lw,-18.6335,-18.223,'--')
    i14[n].fits[2].plot_select('r','r',lw,-18.223,-17.8369)
    i14[n].fits[1].plot_select('l','r',lw,-17.8369,-17.7601)
    i14[n].fits[1].plot_select('l','r',lw,-17.7601,-17.4343,'--')
    i14[n].fits[1].plot_select('l','r',lw,-17.343,-13.76)

    i14[n].fits[1].plot_select('r','#00ff00',lw,-21.4367,-18.6335,'--')
    i14[n].fits[1].plot_select('r','#00ff00',lw,-18.6335,-18.223)
    i14[n].fits[1].plot_select('r','#00ff00',lw,-18.223,-17.7601,'--')
    i14[n].fits[1].plot_select('r','#00ff00',lw,-17.7601,-17.4343)
    i14[n].fits[1].plot_select('r','#00ff00',lw,-17.343,-13.76,'--')

    set_axis(i14[n],-21.4367,-13.76)
    plt.xticks([-20,-15])
    
    plt.axvline(-19.4858,color='k',lw=1)
    plt.axvline(-18.62333,color='k',lw=1)
    plt.axvline(-18.2212,color='k',lw=1)
    plt.axvline(-17.759945,color='k',lw=1)
    plt.axvline(-17.42249,color='k',lw=1)
        
def q_1():
    n = 3
    global j
    j = 3
    init_plot(i14[n])
    i14[n].fits[4].plot_select('l','k',lw,-26.8855,-21.615)
    i14[n].fits[4].plot_select('l','k',lw,-21.615,-18.41,'--')

    i14[n].fits[4].plot_select('r','r',lw,-26.8855,-21.615,'--')
    i14[n].fits[3].plot_select('l','r',lw,-21.615,-20.2475)
    i14[n].fits[3].plot_select('l','r',lw,-20.2475,-18.41,'--')

    i14[n].fits[3].plot_select('r','#00ff00',lw,-26.8855,-20.2475,'--')
    i14[n].fits[3].plot_select('r','#00ff00',lw,-20.2475,-18.41)
    
    set_axis(i14[n],-26.8855,-19.0156)
    plt.xticks([-25,-20])
    
    plt.axvline(-20.2475,color='k',lw=1)
    for k in se14[n][np.abs(se14[n][:,2] - i14[n].fits[4].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
        
def q_2():
    n = 7
    global j
    j = 2
    init_plot(i14[n])
    i14[n].fits[j].plot_select('l','k',lw,-27.85,-21.9669)
    i14[n].fits[j].plot_select('l','k',lw,-21.9669,-21.0443,'--')
    i14[n].fits[j].plot_select('l','k',lw,-21.0443,-19.47)

    i14[n].fits[j].plot_select('r','r',lw,-27.85,-21.9669,'--')
    i14[n].fits[j].plot_select('r','r',lw,-21.9669,-21.0443)
    i14[n].fits[j].plot_select('r','r',lw,-21.0443,-19.47,'--')
    
    set_axis(i14[n],-27.85,-19.47)
    plt.xticks([-25,-20])
    
    for k in se14[n][np.abs(se14[n][:,2] - i14[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)

def q_3():
    n = 8
    global j
    j = 3
    init_plot(i14[n])
    i14[n].fits[0].plot_select('l','k',lw,-32,-30.976)
    i14[n].fits[0].plot_select('l','k',lw,-30.976,-30.0459,'--')
    i14[n].fits[0].plot_select('l','k',lw,-30.0459,-28.8628)
    i14[n].fits[0].plot_select('l','k',lw,-28.8628,-27.5768,'--')
    i14[n].fits[0].plot_select('l','k',lw,-27.5768,-26.6205)
    i14[n].fits[0].plot_select('l','k',lw,-26.6205,-23,'--')

    i14[n].fits[0].plot_select('r','r',lw,-32,-30.976,'--')
    i14[n].fits[0].plot_select('r','r',lw,-30.976,-30.0459)
    i14[n].fits[0].plot_select('r','r',lw,-30.0459,-28.8628,'--')
    i14[n].fits[0].plot_select('r','r',lw,-28.8628,-27.5768)
    i14[n].fits[0].plot_select('r','r',lw,-27.5768,-26.6205,'--')
    i14[n].fits[0].plot_select('r','r',lw,-26.6205,-23)

    set_axis(i14[n],-31.0698,-23)
    plt.xticks([-30,-25])
    
    plt.axvline(-30.9761,color='k',lw=1)
    for k in se14[n][np.abs(se14[n][:,2] - i14[n].fits[3].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
    for k in se14[n][np.abs(se14[n][:,2] - i14[n].fits[0].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
        
def q_4():
    n = 8
    global j
    j = 3
    init_plot(soft14[n])
    i14[n].fits[j].plot_select('l','k',lw,-26.6205,-21.1292)
    i14[n].fits[j].plot_select('l','k',lw,-21.1292,-19.5416,'--')
    i14[n].fits[j].plot_select('l','k',lw,-19.5416,-19.1765)
    i14[n].fits[j].plot_select('l','k',lw,-19.1765,-19.0655,'--')
    i14[n].fits[j].plot_select('l','k',lw,-19.0655,-18.6069)
    i14[n].fits[j].plot_select('l','k',lw,-18.6069,-18.3616,'--')

    i14[n].fits[j].plot_select('r','r',lw,-26.6205,-21.1292,'--')
    i14[n].fits[j].plot_select('r','r',lw,-21.1292,-19.5416)
    i14[n].fits[j].plot_select('r','r',lw,-19.5416,-19.1765,'--')
    i14[n].fits[j].plot_select('r','r',lw,-19.1765,-19.0655)
    i14[n].fits[j].plot_select('r','r',lw,-19.0655,-18.6069,'--')
    i14[n].fits[j].plot_select('r','r',lw,-18.6069,-18.3616)

    set_axis(i14[n],-26.62058,-18.3616)
    plt.xticks([-25,-20])
    
    for k in se14[n][np.abs(se14[n][:,2] - i14[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
    
def q_5():
    n = 0
    global j
    j = 0
    d110 = np.load('2011_0.npy')
    plt.plot(d110[0],d110[1])
    
    i11[n].fits[j].plot_select('l','k',lw,-36.1293,-30.0753)
    i11[n].fits[j].plot_select('l','k',lw,-30.0753,-27.9345,'--')

    i11[n].fits[j].plot_select('r','r',lw,-36.1293,-30.0753,'--')
    i11[n].fits[j].plot_select('r','r',lw,-30.0753,-27.9345)

    set_axis(i11[n],-36.1293,-27.9345)
    plt.xticks([-35,-30])

    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
    
def q_6():
    n = 19
    global j
    j = 1
    init_plot(soft11[n])
    i11[n].fits[j].plot_select('l','k',lw,-35.0326,-31.5792)
    i11[n].fits[j].plot_select('l','k',lw,-31.5792,-25.2446,'--')

    i11[n].fits[j].plot_select('r','r',lw,-35.0326,-31.5792,'--')
    i11[n].fits[j].plot_select('r','r',lw,-31.5792,-25.2446)
    
    set_axis(i11[n],-35.0326,-25.2446)
    plt.xticks([-35,-30])
    
    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)

def p_0():
    n = 7
    global j
    j = 1
    init_plot(soft14[n])
    i14[n].fits[j].plot_select('l','k',lw,-21.02,-14.17)
    i14[n].fits[j].plot_select('l','k',lw,-14.17,-10.565,'--')

    i14[n].fits[j].plot_select('r','r',lw,-21.02,-14.17,'--')
    i14[n].fits[j].plot_select('r','r',lw,-14.17,-10.565)
    
    set_axis(i14[n],-21.02,-10.565)
    plt.xticks([-20,-15])
    
    plt.axvline(-14.17,color='k',lw=1)
    
def p_1():
    n = 9
    global j
    j = 1
    d149 = np.load('2014_9.npy')
    plt.plot(d149[0],d149[1])
    i14[n].fits[j].plot_select('l','k',lw,-19.8524,-13.9762)
    i14[n].fits[j].plot_select('l','k',lw,-13.9762,-6.437,'--')
    i14[n].fits[j].plot_select('r','r',lw,-13.9762,-6.437)
    i14[n].fits[j].plot_select('r','r',lw,-19.8524,-13.9762,'--')
    
    set_axis(i14[n],-19.8524,-8)
    plt.xticks([-15,-10])
    
    plt.axvline(-13.7033,color='k',lw=1)
    
def p_2():
    n = 0
    global j
    j = 1
    init_plot(i11[n])

    i11[n].fits[j].plot_select('l','k',lw,-42.9385,-37.5519)
    i11[n].fits[j].plot_select('l','k',lw,-37.5519,-37.1954,'--')
    i11[n].fits[j].plot_select('l','k',lw,-37.1954,-36.8603)
    i11[n].fits[j].plot_select('l','k',lw,-36.8603,-36.4394,'--')
    i11[n].fits[j].plot_select('l','k',lw,-36.4394,-36.143)
    i11[n].fits[j].plot_select('l','k',lw,-36.143,-30.0926,'--')

    i11[n].fits[j].plot_select('r','r',lw,-42.9385,-37.5519,'--')
    i11[n].fits[j].plot_select('r','r',lw,-37.5519,-37.1954)
    i11[n].fits[j].plot_select('r','r',lw,-37.1954,-36.8603,'--')
    i11[n].fits[j].plot_select('r','r',lw,-36.8603,-36.4394)
    i11[n].fits[j].plot_select('r','r',lw,-36.4394,-36.143,'--')
    i11[n].fits[j].plot_select('r','r',lw,-36.143,-30.0926)

    set_axis(i11[n],-42.9385,-30.0926)
    plt.xticks([-40,-35])

    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
        
def p_3():
    n = 1
    global j
    j = 0
    init_plot(soft11[n])
    i11[n].fits[j].plot_select('l','k',lw,-26.3665,-20.087)
    i11[n].fits[j].plot_select('l','k',lw,-20.087,-17.45,'--')

    i11[n].fits[j].plot_select('r','r',lw,-26.3665,-20.087,'--')
    i11[n].fits[j].plot_select('r','r',lw,-20.087,-17.45)
    
    set_axis(i11[n],-26.3665,-17.45)
    plt.ylim([11.35,11.89])
    plt.xticks([-25,-20])

    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
                
def p_4():
    n = 3
    global j
    j = 0
    init_plot(soft11[n])
    i11[n].fits[j].plot_select('l','k',lw,-40.7668,-35.4962)
    i11[n].fits[j].plot_select('l','k',lw,-35.4962,-29.9666,'--')

    i11[n].fits[j].plot_select('r','r',lw,-40.7668,-35.4962,'--')
    i11[n].fits[j].plot_select('r','r',lw,-35.4962,-29.9666)
    set_axis(i11[n],-40.7668,-29.9666)
    plt.xticks([-40,-35])

    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)

def p_5():
    n = 6
    global j
    j = 0
    init_plot(soft11[n])
    i11[n].fits[j].plot_select('l','k',lw,-45.2278,-35.6776)
    i11[n].fits[j].plot_select('l','k',lw,-35.6776,-30.5427,'--')

    i11[n].fits[j].plot_select('r','r',lw,-45.2278,-35.6776,'--')
    i11[n].fits[j].plot_select('r','r',lw,-35.6776,-30.5427)
    
    set_axis(i11[n],-42.174,-30.5427)
    plt.xticks([-40,-35])

    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)

def p_6():
    n = 8
    global j 
    j = 0
    init_plot(i11[n])
    i11[n].fits[j].plot_select('l','k',lw,-40.4377,-35.138)
    i11[n].fits[j].plot_select('l','k',lw,-35.138,-34.0386,'--')
    i11[n].fits[j].plot_select('l','k',lw,-34.0386,-33.4822)
    i11[n].fits[j].plot_select('l','k',lw,-33.4822,-30.6139,'--')
    i11[n].fits[j].plot_select('r','r',lw,-40.4377,-35.138,'--')
    i11[n].fits[j].plot_select('r','r',lw,-35.138,-34.0386)
    i11[n].fits[j].plot_select('r','r',lw,-34.0386,-33.4822,'--')
    i11[n].fits[j].plot_select('r','r',lw,-33.4822,-30.6139)
    
    set_axis(i11[n],-40.4377,-30.6139)
    plt.xticks([-40,-35])
    
    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
    
def p_7():
    n = 14
    global j 
    j = 1
    init_plot(i11[n])
    i11[n].fits[j].plot_select('l','k',lw,-41.6623,-34.0011)
    i11[n].fits[j].plot_select('l','k',lw,-34.0011,-30.424,'--')
    i11[n].fits[j].plot_select('r','r',lw,-41.6623,-34.0011,'--')
    i11[n].fits[j].plot_select('r','r',lw,-34.0011,-30.424)
    
    set_axis(i11[n],-41.6623,-30.424)
    plt.xticks([-40,-35])
    
    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
      
def p_8():
    n = 15
    global j 
    j = 1
    init_plot(i11[n])

    i11[n].fits[j].plot_select('l','k',lw,-42.7665,-34.0758)
    i11[n].fits[j].plot_select('l','k',lw,-34.0758,-30.0193,'--')

    i11[n].fits[j].plot_select('r','r',lw,-42.7665,-34.07587,'--')
    i11[n].fits[j].plot_select('r','r',lw,-34.0758,-30.0193)

    set_axis(i11[n],-42.7665,-30.307)
    plt.xticks([-40,-35])

    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)

def p_9():
    n = 19
    global j 
    j = 2
    init_plot(i11[n])
    i11[n].fits[j].plot_select('l','k',lw,-41.1981,-35.3895)
    i11[n].fits[j].plot_select('l','k',lw,-35.3895,-31.6231,'--')

    i11[n].fits[j].plot_select('r','r',lw,-41.1981,-35.3895,'--')
    i11[n].fits[j].plot_select('r','r',lw,-35.3895,-31.6231)
    
    set_axis(i11[n],-41.1981,-31.6231)
    plt.xticks([-40,-35])

    for k in se11[n][np.abs(se11[n][:,2] - i11[n].fits[j].delta_phi)<0.01]:
        plt.axvline(k[0],color='k',lw=1)
   
i11 = Analysis.Run(np.load('2011_Raw.npy'), np.load('2011_Prune_strong_flat.npy'), Analysis.bval2011, np.load('2011bf_new_prune_flat_edit.npy')).prune_data
i14 = Analysis.Run(np.load('2014_Raw.npy'), np.load('2014_Prune_strong.npy'), Analysis.bval2014, np.load('2014bf_new_prune_edit.npy')).prune_data
soft11 = Analysis.Run(np.load('2011_Raw.npy'), np.load('2011_Prune_new_flat.npy'), Analysis.bval2011, np.load('2011bf_new_prune_flat_edit.npy')).prune_data
soft14 = Analysis.Run(np.load('2014_Raw.npy'), np.load('2014_Prune_new.npy'), Analysis.bval2014, np.load('2014bf_new_prune_edit.npy')).prune_data
se11 = np.load('2011se_flat_edit.npy')
se14 = np.load('2014se_edit.npy')
d140 = np.load('2014_0.npy')
d144 = np.load('2014_4.npy')

q = [q_0,q_1,q_2,q_3,q_4] # Only use 2014 pi/4 events ,q_5,q_6] 
p = [p_0,p_1,p_2,p_4,p_5,p_6,p_7,p_8,p_9]

q_perm = [perm for perm in itertools.combinations(q, 3)]
p_perm = [perm for perm in itertools.combinations(p, 3)]

#Base figure skeleton:

#Full and windows all permutations

mpl.rcParams.update({'font.size': 8})

counter = 0

gs = GridSpec(1000,1000)
num_y = 6
num_x = 2
right_space = 0
mid_space = 60
bot_space = 25
y_len = int((1000-(num_y-1)*bot_space)/num_y)
x_len = int((1000-right_space-(num_x-1)*mid_space)/num_x)
ypad = 2
xpad = 4
lw = 3

for q_plots in q_perm:
    for p_plots in p_perm:

        fig = plt.figure(figsize=[7.2,7.2*1.2])

        #-------------------------------------------------------------------------------
        ax1 = plt.subplot(gs[3*y_len+3*bot_space:4*y_len+3*bot_space,right_space:2*x_len+right_space+mid_space])
        ax1.annotate('G', xy=(0.01, 0.88), xycoords="axes fraction")
        n = 0
        j = 3
        plt.plot(d140[0],d140[1])
        i14[n].fits[3].plot_select('l','k',lw,-31,-22.1535)
        i14[n].fits[3].plot_select('l','k',lw,-22.1535,-19.8804,'--')

        i14[n].fits[3].plot_select('r','r',lw,-31,-22.1535,'--')
        i14[n].fits[3].plot_select('r','r',lw,-22.1535,-19.8804)

        i14[n].fits[2].plot_select('l','r',lw,-19.8804,-19.4419)
        i14[n].fits[2].plot_select('l','r',lw,-19.4419,-19.0109,'--')

        i14[n].fits[2].plot_select('r','#00ff00',lw,-20.99,-19.4419,'--')
        i14[n].fits[2].plot_select('r','#00ff00',lw,-19.4419,-18.675)
        i14[n].fits[1].plot_select('l','#00ff00',lw,-18.675,-14.87)
        i14[n].fits[1].plot_select('l','#00ff00',lw,-14.87,-10.93,'--')

        i14[n].fits[1].plot_select('r','orange',lw,-18.675,-14.87,'--')
        i14[n].fits[1].plot_select('r','orange',lw,-14.87,-11.7825)
        i14[n].fits[0].plot_select('l','orange',lw,-11.7825,-8.417)
        i14[n].fits[0].plot_select('l','orange',lw,-8.417,-6.98202,'--')

        i14[n].fits[0].plot_select('r','k',lw,-10.8774,-8.417,'--')
        i14[n].fits[0].plot_select('r','k',lw,-8.417,-6.98202)

        for k in se14[n]:
            ax1.axvline(k[0],color='k',lw=1)

        set_axis(i14[n],-31,-6.98202)
        plt.xticks([-30,-20,-10])

        ax2 = plt.subplot(gs[4*y_len+4*bot_space:5*y_len+4*bot_space,right_space:2*x_len+right_space+mid_space])
        ax2.annotate('H', xy=(0.01, 0.88), xycoords="axes fraction")
        n = 4
        j = 1
        plt.plot(d144[0],d144[1])
        i14[n].fits[1].plot_select('l','k',lw,-31,-21.7726)
        i14[n].fits[1].plot_select('l','k',lw,-21.7726,-17.7358,'--')

        i14[n].fits[1].plot_select('r','r',lw,-31,-21.7726,'--')
        i14[n].fits[1].plot_select('r','r',lw,-21.7726,-17.7358)

        i14[n].fits[0].plot_select('l','r',lw,-17.7358,-14.4199)
        i14[n].fits[0].plot_select('l','r',lw,-14.4199,-7.19,'--')

        i14[n].fits[0].plot_select('r','#00ff00',lw,-17.7358,-14.4199,'--')
        i14[n].fits[0].plot_select('r','#00ff00',lw,-14.4199,-7.19)

        for k in se14[n]:
            ax2.axvline(k[0],color='k',lw=1)

        set_axis(i14[n],-31,-7.19)
        plt.xticks([-30,-20,-10])

        ax3 = plt.subplot(gs[5*y_len+5*bot_space:6*y_len+5*bot_space,right_space:2*x_len+right_space+mid_space])
        ax3.annotate('I', xy=(0.01, 0.88), xycoords="axes fraction")
        n = 5
        j = 1
        soft = Analysis.Run(np.load('2011_Raw.npy'), np.load('2011_Prune_new_flat.npy'), Analysis.bval2011, np.load('2011bf_new_prune_flat_edit.npy')).prune_data
        init_plot(soft[n])
        i11[n].fits[1].plot_select('l','k',lw,-44,-35.2168)
        i11[n].fits[1].plot_select('l','k',lw,-35.2168,-32.76,'--')

        i11[n].fits[1].plot_select('r','r',lw,-44,-35.2168,'--')
        i11[n].fits[1].plot_select('r','r',lw,-35.2168,-34.0052)

        i11[n].fits[0].plot_select('l','r',lw,-34.0052,-30.2804)
        i11[n].fits[0].plot_select('l','r',lw,-30.2804,-10,'--')

        i11[n].fits[0].plot_select('r','#00ff00',lw,-34.0052,-30.2804,'--')
        i11[n].fits[0].plot_select('r','#00ff00',lw,-30.2804,-10)

        for k in se11[n]:
            ax3.axvline(k[0],color='k',lw=1)

        set_axis(i11[n],-43.2812,-21.5518)
        plt.xticks([-40,-30])

        #PI--------------------------------------------

        ax4 = plt.subplot(gs[:y_len,right_space:x_len+right_space])
        ax4.annotate('A', xy=(0.01, 0.88), xycoords="axes fraction")
        p_plots[0]()

        ax5 = plt.subplot(gs[y_len+bot_space:2*y_len+bot_space,right_space:x_len+right_space])
        ax5.annotate('B', xy=(0.01, 0.88), xycoords="axes fraction")
        p_plots[1]()

        ax6 = plt.subplot(gs[2*y_len+2*bot_space:3*y_len+2*bot_space,right_space:x_len+right_space])
        ax6.annotate('C', xy=(0.01, 0.88), xycoords="axes fraction")
        p_plots[2]()

        #PI/4---------------------------------------------------

        ax7 = plt.subplot(gs[:y_len,x_len+mid_space+right_space:2*x_len+right_space+mid_space])
        ax7.annotate('D', xy=(0.01, 0.88), xycoords="axes fraction")
        q_plots[0]()

        ax8 = plt.subplot(gs[y_len+bot_space:2*y_len+bot_space,x_len+mid_space+right_space:2*x_len+right_space+mid_space])
        ax8.annotate('E', xy=(0.01, 0.88), xycoords="axes fraction")
        q_plots[1]()

        ax9 = plt.subplot(gs[2*y_len+2*bot_space:3*y_len+2*bot_space,x_len+mid_space+right_space:2*x_len+right_space+mid_space])
        ax9.annotate('F', xy=(0.01, 0.88), xycoords="axes fraction")
        q_plots[2]()

        ax1.set_ylabel('R$_{D}$ (k$\Omega$)',size=12)
        ax1.yaxis.set_label_coords(-.05, 1.1)

        ax3.set_xlabel('V$_{P}$ (mV)',size=12)
        
        fig.savefig(dirpath+'/Permutations/'+'Fig_Permutation_{}.png'.format(counter), format="PNG")

        plt.close(fig)
        counter += 1
