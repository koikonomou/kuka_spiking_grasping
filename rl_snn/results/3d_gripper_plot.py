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
def plot_joints(data,env_num):
    px = []
    py =[]
    pz = []
    status = []
    time = []
    path = data["path"]
    trials =len(data['path'][env_num])
    final_state = data["final_state"][env_num]
    for trial in range(trials):
        px.append( path[env_num][trial][15] )
        py.append( path[env_num][trial][16] )
        pz.append( path[env_num][trial][17] )
    
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

directory = '../record_data/snn_subgoal_revision/actions/20_envs'
ep_number = 0
run_num = 0
file = 'episode_'+'{ep_num}'.format(ep_num=ep_number) + '_run_'+'{run}'.format(run=run_num)+'.p'
data = pickle.load(open(directory+'/'+file, 'rb'))
x,y,z,time,status = plot_joints(data=data,env_num=5)

a = []
b = []
c = []
for item in x:
    a.append(float(item))
for item in y:
    b.append(float(item))
for item in z:
    c.append(float(item))

r = np.array(a)
s = np.array(b)
t = np.array(c)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(2,2,2, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
# ax.set_zlabel("z axis")
ax.set_title('hybrid-DDPG sub-goals \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
ax.scatter(r,s, zs = t, c= color, s=80, cmap='binary' )
ax.plot3D(r,s,z,c= color_line )
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])


directory = '../record_data/snn_simple_revision/actions/20_envs'
ep_number = 0
run_num = 0
file = 'episode_'+'{ep_num}'.format(ep_num=ep_number) + '_run_'+'{run}'.format(run=run_num)+'.p'
data = pickle.load(open(directory+'/'+file, 'rb'))
x,y,z,time,status = plot_joints(data=data,env_num=5)

a = []
b = []
c = []
for item in x:
    a.append(float(item))
for item in y:
    b.append(float(item))
for item in z:
    c.append(float(item))

r = np.array(a)
s = np.array(b)
t = np.array(c)

ax = fig.add_subplot(2,2,1, projection='3d')
ax.set_xlabel("x axis")

ax.set_ylabel("y axis ")
# ax.set_zlabel("z axis")
ax.set_title('hybrid-DDPG end-to-end \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
ax.scatter(r, s, zs = t, c= color,s=80, cmap='binary' )

ax.plot3D(r, s, z, c= color_line)
# plt.axis('off')
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])


# plt.show()

def _actions(data,env_num):
    px = []
    py =[]
    pz = []
    status = []
    time = []
    action_vals = []
    suc_rate = 0
    path = data["path"]
    actions = data ["actions"]
    trials = len(data['path'][env_num])
    final_state = data["final_state"][env_num]
    # action_vals.append(data["actions"])
    if final_state == 1:
        status.append('Success')
        time.append(data["time"][env_num])
        suc_rate +=1
    if final_state == 2:
        status.append("Collision")
        time.append(data["time"][env_num])
    if final_state == 3:
        status.append("Time out")
        time.append(data["time"][env_num])
    time = str(round(np.sum(time), 3))
    return actions,time,status,suc_rate


def _sum(arr):
    sum = 0
    for i in arr:
        i = float(i)
        sum = sum + i
    if sum != 0:
        t = sum / trials
    else:
        t = 0
    return t

def _plot_subgoal(trials,env_n,count, graph):
    directory = '../record_data/snn_simple_revision/actions/20_envs'
    f_trials = []
    f_trials_av_time = []
    suc_rate_av = []
    count_fig = count
    for i in range(trials):
        file = 'episode_'+'{ep_num}'.format(ep_num=ep_number) + '_run_'+'{run}'.format(run=i)+'.p'
        data = pickle.load(open(directory+'/'+file, 'rb'))
        act, time, stat, suc_rate = _actions(data, env_n)
        f_trials.append(act)
        f_trials_av_time.append(time)
        suc_rate_av.append(suc_rate)

    trial_av_time = _sum(f_trials_av_time)
    suc_rate = _sum(suc_rate_av)

    p = 1
    ax = fig.add_subplot(graph, 2, count_fig)
    # Create timeseries ja1
    for i in range(trials):

        fl_time = float(f_trials_av_time[i])
        len_x = len(f_trials[i])

        timeseries_ar = np.linspace(0,fl_time,len_x)
        timeseries = np.round(np.array(timeseries_ar), 2)
        x = np.round(np.array(f_trials[i]), 2)
        # ax.plot(timeseries, x)
        # p +=1
    for i in range (1,6):
        ax.plot(np.NaN, np.NaN, label='j'+str(i))
    plt.axis([0, 1.05, -3.5, 1.5])
    plt.legend(loc="upper left")
    # print(act)
    # JOINT 
    # ax.set_ylabel("Joints ")
    # ax.set_ylabel('hybrid DDPG simple  \nexec time:%s \nstatus: %s'% (np.round(trial_av_time,3) , np.round(suc_rate,3)), fontsize=7)






def _plot_simple(trials,env_n,count,graph):
    count_fig = count
    directory = '../record_data/snn_subgoal_revision/actions/20_envs'
    f_trials = []
    f_trials_av_time = []
    suc_rate_av = []
    for i in range(trials):
        file = 'episode_'+'{ep_num}'.format(ep_num=i) + '_run_'+'{run}'.format(run=i)+'.p'
        data = pickle.load(open(directory+'/'+file, 'rb'))
        act, time, stat, suc_rate = _actions(data, env_n)
        f_trials.append(act)
        f_trials_av_time.append(time)
        suc_rate_av.append(suc_rate)

    trial_av_time = _sum(f_trials_av_time)
    suc_rate = _sum(suc_rate_av)

    ax = fig.add_subplot(graph, 2, count_fig)
    plt.axis([0, 3.7, -3.5, 1.5])
    
    # Create timeseries ja1
    for i in range(trials):
        fl_time = float(f_trials_av_time[i])
        len_x = len(f_trials[i])
        timeseries_ar = np.linspace(0,fl_time,len_x)
        timeseries = np.round(np.array(timeseries_ar), 2)
        x = np.round(np.array(f_trials[i]), 2)
        # print(len(x[0]))
        ax.plot(timeseries, x)

    # ax.set_ylabel("Joints ")
    # ax.set_ylabel('hybrid DDPG subgoal \nexec time:%s \nstatus: %s'% (np.round(trial_av_time,3) , np.round(suc_rate,3)), fontsize=7)


ep_number = 0
run_num = 0
env_n = 5
trials = 1
graph = 2
# fig = plt.figure()

_plot_simple(trials,env_n,4,graph)
_plot_subgoal(trials,env_n,3,graph)
# _plot_simple(trials,5,3,graph)
# _plot_subgoal(trials,5,4,graph)
# _plot_simple(trials,4,5,graph)
# _plot_subgoal(trials,4,6,graph)
fig.subplots_adjust(hspace=0)
plt.show()