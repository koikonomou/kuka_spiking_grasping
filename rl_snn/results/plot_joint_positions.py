import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from colorspacious import cspace_converter

color= mcolors.XKCD_COLORS['xkcd:pumpkin']
color_line= mcolors.XKCD_COLORS['xkcd:browny orange']

directory = '../record_data/snn_simple_revision/50_trials'
ep_number=0
run_num = 0
env_n = 0
file = 'episode_'+'{ep_num}'.format(ep_num=ep_number) + '_run_'+'{run}'.format(run=run_num)+'.p'
data = pickle.load(open(directory+'/'+file, 'rb'))


def plot_joints(data,env_num,n1,n2,n3):
    px = []
    py =[]
    pz = []
    status = []
    time = []
    path = data["path"]
    trials =len(data['path'][env_num])
    final_state = data["final_state"][env_num]
    for trial in range(trials):
        px.append( path[env_num][trial][n1] )
        py.append( path[env_num][trial][n2] )
        pz.append( path[env_num][trial][n3] )
    
    if final_state == 1:
        status.append('Success')
        time.append(data["time"][env_num])
    if final_state == 2:
        status.append("Collision")
        time.append(data["time"][env_num])
    if final_state == 3:
        status.append("Time out")
        time.append(data["time"][env_num])
    time = str(round(np.sum(time), 3))
    return px,py,pz,time,status





# Joint a1 = 0,1,2
x_a1,y_a1,z_a1,time_a1,status_a1 = plot_joints(data=data, env_num=env_n, n1=0, n2=1, n3=2)
# Joint a2 = 3,4,5
x_a2,y_a2,z_a2,time_a2,status_a2 = plot_joints(data=data, env_num=env_n, n1=3, n2=4, n3=5)
# Joint a3 = 6,7,8
x_a3,y_a3,z_a3,time_a3,status_a3 = plot_joints(data=data, env_num=env_n, n1=6, n2=7, n3=8)
# Joint a4 = 9,10,11
x_a4,y_a4,z_a4,time_a4,status_a4 = plot_joints(data=data, env_num=env_n, n1=9, n2=10, n3=11)
# Joint a5 = 12,13,14
x_a5,y_a5,z_a5,time_a5,status_a5 = plot_joints(data=data, env_num=env_n, n1=12, n2=13, n3=14)
# Joint a6 = 15,16,17
x_a6,y_a6,z_a6,time_a6,status_a6 = plot_joints(data=data, env_num=env_n, n1=15, n2=16, n3=17)

# Create timeseries ja1
fl_time_a1 = float(time_a1)
len_x_a1 = len(x_a1)
timeseries_ar_a1=np.linspace(0,fl_time_a1,len_x_a1)

timeseries_a1 = np.round(np.array(timeseries_ar_a1), 2)
x_a1 = np.round(np.array(x_a1), 2)
y_a1 = np.round(np.array(y_a1), 2)
z_a1 = np.round(np.array(z_a1), 2)

# Create timeseries ja2
fl_time_a2 = float(time_a2)
len_x_a2 = len(x_a2)
timeseries_ar_a2=np.linspace(0,fl_time_a2,len_x_a2)

timeseries_a2 = np.round(np.array(timeseries_ar_a2), 2)
x_a2 = np.round(np.array(x_a2), 2)
y_a2 = np.round(np.array(y_a2), 2)
z_a2 = np.round(np.array(z_a2), 2)

# Create timeseries ja3
fl_time_a3 = float(time_a3)
len_x_a3 = len(x_a3)
timeseries_ar_a3=np.linspace(0,fl_time_a3,len_x_a3)

timeseries_a3 = np.round(np.array(timeseries_ar_a3), 2)
x_a3 = np.round(np.array(x_a3), 2)
y_a3 = np.round(np.array(y_a3), 2)
z_a3 = np.round(np.array(z_a3), 2)

# Create timeseries ja4
fl_time_a4 = float(time_a4)
len_x_a4 = len(x_a4)
timeseries_ar_a4=np.linspace(0,fl_time_a4,len_x_a4)

timeseries_a4 = np.round(np.array(timeseries_ar_a4), 2)
x_a4 = np.round(np.array(x_a4), 2)
y_a4 = np.round(np.array(y_a4), 2)
z_a4 = np.round(np.array(z_a4), 2)

# Create timeseries ja5
fl_time_a5 = float(time_a5)
len_x_a5 = len(x_a5)
timeseries_ar_a5=np.linspace(0,fl_time_a5,len_x_a5)

timeseries_a5 = np.round(np.array(timeseries_ar_a5), 2)
x_a5 = np.round(np.array(x_a5), 2)
y_a5 = np.round(np.array(y_a5), 2)
z_a5 = np.round(np.array(z_a5), 2)

# Create timeseries ja6
fl_time_a6 = float(time_a6)
len_x_a6 = len(x_a6)
timeseries_ar_a6=np.linspace(0,fl_time_a6,len_x_a6)

timeseries_a6 = np.round(np.array(timeseries_ar_a6), 2)
x_a6 = np.round(np.array(x_a6), 2)
y_a6 = np.round(np.array(y_a6), 2)
z_a6 = np.round(np.array(z_a6), 2)




fig = plt.figure()
# plt.imshow(a, cmap='hot', interpolation='nearest')
# ax = fig.add_subplot(1,3,3,)
# heatmap, xedges, yedges = np.histogram2d( timeseries_a1,z_a1, bins=100)
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


# JOINT a1
ax = fig.add_subplot(6, 3, 1)
ax.plot(timeseries_a1, x_a1)
ax.set_xlabel("x axis movement ")
ax.set_ylabel("Joint a1 ")
ax = fig.add_subplot(6, 3, 2)
ax.plot(timeseries_a1, y_a1)
ax.set_xlabel("y axis movement")
ax = fig.add_subplot(6, 3, 3)
ax.plot(timeseries_a1, z_a1)
ax.set_xlabel("z axis movement")


# JOINT a2
ax = fig.add_subplot(6, 3, 4)
ax.plot(timeseries_a2, x_a2)
ax.set_xlabel("x axis movement ")
ax.set_ylabel("Joint a2 ")
ax = fig.add_subplot(6, 3, 5)
ax.plot(timeseries_a2, y_a2)
ax.set_xlabel("y axis movement")
ax = fig.add_subplot(6, 3, 6)
ax.plot(timeseries_a2, z_a2)
ax.set_xlabel("z axis movement")



# JOINT a3
ax = fig.add_subplot(6, 3, 7)
ax.plot(timeseries_a3, x_a3)
ax.set_xlabel("x axis movement ")
ax.set_ylabel("Joint a3 ")
ax = fig.add_subplot(6, 3, 8)
ax.plot(timeseries_a3, y_a3)
ax.set_xlabel("y axis movement")
ax = fig.add_subplot(6, 3, 9)
ax.plot(timeseries_a3, z_a3)
ax.set_xlabel("z axis movement")


# JOINT a4
ax = fig.add_subplot(6, 3, 10)
ax.plot(timeseries_a4, x_a4)
ax.set_xlabel("x axis movement ")
ax.set_ylabel("Joint a4 ")
ax = fig.add_subplot(6, 3, 11)
ax.plot(timeseries_a4, y_a4)
ax.set_xlabel("y axis movement")
ax = fig.add_subplot(6, 3, 12)
ax.plot(timeseries_a4, z_a4)
ax.set_xlabel("z axis movement")

# JOINT a5
ax = fig.add_subplot(6, 3, 13)
ax.plot(timeseries_a5, x_a5)
ax.set_xlabel("x axis movement ")
ax.set_ylabel("Joint a5 ")
ax = fig.add_subplot(6, 3, 14)
ax.plot(timeseries_a5, y_a5)
ax.set_xlabel("y axis movement")
ax = fig.add_subplot(6, 3, 15)
ax.plot(timeseries_a5, z_a5)
ax.set_xlabel("z axis movement")

# JOINT a6
ax = fig.add_subplot(6, 3, 16)
ax.plot(timeseries_a6, x_a6)
ax.set_xlabel("x axis movement ")
ax.set_ylabel("Joint a6 ")
ax = fig.add_subplot(6, 3, 17)
ax.plot(timeseries_a6, y_a6)
ax.set_xlabel("y axis movement")
ax = fig.add_subplot(6, 3, 18)
ax.plot(timeseries_a6, z_a6)
ax.set_xlabel("z axis movement")



plt.show()

