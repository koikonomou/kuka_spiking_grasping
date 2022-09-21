import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from utility import analyze_run, plot_robot_paths,plot_results

MODEL_NAME = 'sddpg_bw_5'
# MODEL_NAME = 'ddpg'
# MODEL_NAME = 'ddpg_poisson'
FILE_NAME = MODEL_NAME + '_0_199.p'

run_data = pickle.load(open('../record_data/2022_09_21-07_16_42_PM.p', 'rb'))
plot_results(run_data)
s_list, p_dis, p_time, p_spd = analyze_run(run_data)
print(MODEL_NAME + " random simulation results:")
print("Success: ", s_list[0], " Collision: ", s_list[1], " Overtime: ", s_list[2])
print("Average Path Distance of Success Routes: ", np.mean(p_dis[p_dis > 0]), ' m')
print("Average Path Time of Success Routes: ", np.mean(p_time[p_dis > 0]), ' s')
# print("Average Path Speed of Success Routes: ", np.mean(p_spd[p_dis > 0]), ' m/s')


init_pos_pos = [[0.0, -1.35, 1.9, 0.0, 0.61],[0.0, -2.5, 2.3, 0.0, 1.0],[0.0, -2.0, 1.5, 0.0, 1.55], [0.0, -1.57, 0.6, 0.0, 2.09]]
start_goal_pos = [0.5, 0.0, 0.85]

goal_list = start_goal_pos[1]
plot_robot_paths(run_data, goal_list)
plt.show()