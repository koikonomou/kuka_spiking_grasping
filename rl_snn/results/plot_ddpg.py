import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from astropy.table import QTable, Table, Column
from astropy import units as u
sys.path.append('../..')
from utility import analyze_run, plot_robot_paths,plot_results
import math

directory = '../record_data/snn_subgoal_revision/actions/20_envs'

suc_rate =[]
time = []
trial = []
counter1 = 1
d = 0
average_time = []
for i in range(5):
    run_num = i
    file = 'episode_'+'{ep_num}'.format(ep_num=i) + '_run_'+'{run}'.format(run=run_num)+'.p'
    data = pickle.load(open(directory+'/'+file, 'rb'))
    run = len(data["final_state"])
    for r in range(run):
        if data["final_state"][r] == 1:
            trial.append(1)
            d =d+1
            counter1 = counter1+1
            if math.isnan(data["time"][r]) or math.isinf(data["time"][r]):
                counter= counter-1
                d=d-1
                trial.pop()
            else:
                time.append(data["time"][r])
        if data["final_state"][r] == 2:
            trial.append(0)
            d =d+1
            # time.appcnd(data["time"][r])
        if data["final_state"][r] == 3:
            trial.append(0)
            d =d+1
            # time.append(data["time"][r])
    ave_rew = np.sum(trial,axis=0)/d
    trial.clear()
    suc_rate.append(ave_rew)
    sum_time = np.sum(time,axis=0)/counter1
    time.clear()
    d=0
    counter1 = 1
    average_time.append(sum_time)
y1 = suc_rate
x1 = average_time
print("Average success over 50 trials:", sum(suc_rate)/50)
print("Average execution time over 50 trials:", sum(average_time)/len(suc_rate))
fig, ax = plt.subplots()
ax.plot(x1, y1, 'o', color='b')

dir2 = '../record_data/snn_simple_revision/actions/20_envs'
suc_rate2 =[]
time2 = []
trial2 = []
counter = 1
c = 0
average_time2 = []
for i in range(5):
    run_num = i
    file = 'episode_'+'{ep_num}'.format(ep_num=0) + '_run_'+'{run}'.format(run=run_num)+'.p'
    data = pickle.load(open(dir2+'/'+file, 'rb'))
    run = len(data["final_state"])
    for r in range(run):
        if data["final_state"][r] == 1:
            trial2.append(1)
            counter = counter +1
            c=c+1
            if math.isnan(data["time"][r]) or math.isinf(data["time"][r]):
                counter= counter-1
                c=c-1
                trial2.pop()
            else:
                time2.append(data["time"][r])
                print(data["time"][r])
        if data["final_state"][r] == 2:
            trial2.append(0)
            c=c+1
            # time2.append(data["time"][r])
        if data["final_state"][r] == 3:
            trial2.append(0)
            c=c+1
            # time2.append(data["time"][r])
    if trial != 0:
        ave_rew = np.sum(trial2,axis=0)/c
    else:
        ave_rew = 0
    trial2.clear()
    c=0
    suc_rate2.append(ave_rew)

    sum_time = np.sum(time2,axis=0)/counter
    time2.clear()
    counter = 1
    average_time2.append(sum_time)
y2 = suc_rate2
x2 = average_time2
print("Average success over 50 trials:", sum(suc_rate2)/50)
print("Average execution time over 50 trials:", sum(average_time2)/len(suc_rate2))
# fig, ax = plt.subplots()
# ax.plot(x1, y1, 'o', color='b')
ax.plot(x2, y2, 'o', color='r')
ax.set_ylabel('Success rate for each trial (5trials)')
ax.set_xlabel('Average execution time for each trial')
ax.legend(labels=['hubrid-DDPG', 'DDPG'])


t = Table([y1, y2, x1, x2], names=('Success rate SNN', 'Success rate DDPG', 'Average execution time SNN', 'Average execution time DDPG'))
# print(*t)

# plt.show()

import xlsxwriter

workbook = xlsxwriter.Workbook('subgoals-revision-v1.xlsx')
worksheet = workbook.add_worksheet()
# array = [["SNN_without Succ rate"],["snn_without exec time"],["SNN with Succ rate"],["SNN with exec time"]]
array = []
array.append(y1)
array.append(x1)
array.append(y2)
array.append(x2)
row = 0

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)

workbook.close()