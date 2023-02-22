import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mycolorpy import colorlist as mcp
from colorspacious import cspace_converter

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


    ax = fig.add_subplot(graph, 2, count_fig)
    # Create timeseries ja1
    for i in range(trials):

        fl_time = float(f_trials_av_time[i])
        len_x = len(f_trials[i])
        timeseries_ar = np.linspace(0,fl_time,len_x)
        timeseries = np.round(np.array(timeseries_ar), 2)
        x = np.round(np.array(f_trials[i]), 2)
        ax.plot(timeseries, x)
    plt.axis([0, 1.05, -3.5, 1.5])

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
graph = 1
fig = plt.figure()

_plot_simple(trials,env_n,2,graph)
_plot_subgoal(trials,env_n,1,graph)
# _plot_simple(trials,5,3,graph)
# _plot_subgoal(trials,5,4,graph)
# _plot_simple(trials,4,5,graph)
# _plot_subgoal(trials,4,6,graph)

plt.show()
