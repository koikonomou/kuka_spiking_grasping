import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../..')
from utility import analyze_run, plot_robot_paths,plot_results

# This code generate the graphs similar to the paper "Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research"
plt.style.use('_mpl-gallery')
directory = '../record_data/snn_without_subgoal/5_10_b'
data = []

suc_rate = []
median = []
upper_value = []
lower_value = []
env = []
trial = []

for e in range(1,25):
	episode = e
	r=0
	for i in range(5):
		run_num = i
		file = 'episode_'+'{ep_num}'.format(ep_num=episode) + '_run_'+'{run}'.format(run=run_num)+'.p'
		data = pickle.load(open(directory+'/'+file, 'rb'))
		run = len(data["final_state"])
		env.clear()
		for r in range(run):
			if data["final_state"][r] == 1:
				env.append(1)
			else :
				env.append(0)
		trial.append(env)
	sum_values=np.sum(trial,axis=0)
	sum_values=sum_values[1:6]/5
	Q1 = np.percentile(sum_values[1:6], 25, interpolation = 'midpoint')
	Q3 = np.percentile(sum_values[1:6], 75, interpolation = 'midpoint')
	m = sum_values[3]
	trial.clear()

	upper_value.append(Q1)
	lower_value.append(Q3)
	median.append(m)

x = np.linspace(0, 10, 24)
y1 = upper_value
y2 = lower_value

fig, ax = plt.subplots()
ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
ax.plot(x, median, linewidth=2)

plt.show()

# dir2 = '../record_data/snn_with_subgoals/5_10_b'
# median1 = []
# upper_value1 = []
# lower_value1 = []
# env1 = []
# trial1 =[]
# data1=[]
# for e in range(1,25):
# 	episode = e
# 	r=0
# 	for i in range(5):
# 		run_num = i
# 		file = 'episode_'+'{ep_num}'.format(ep_num=episode) + '_run_'+'{run}'.format(run=run_num)+'.p'
# 		data1 = pickle.load(open(dir2+'/'+file, 'rb'))
# 		run = len(data1["final_state"])
# 		env1.clear()
# 		for r in range(run):
# 			if data1["final_state"][r] == 1:
# 				env1.append(1)
# 			else :
# 				env1.append(0)
# 		trial1.append(env1)
# 	sum_values=np.sum(trial1,axis=0)
# 	sum_values=sum_values[1:6]/5
# 	Q1 = np.percentile(sum_values[1:6], 25, interpolation = 'midpoint')
# 	Q3 = np.percentile(sum_values[1:6], 75, interpolation = 'midpoint')
# 	m = sum_values[3]
# 	trial1.clear()

# 	upper_value1.append(Q1)
# 	lower_value1.append(Q3)
# 	median1.append(m)

# x2 = np.linspace(0, 10, 24)
# y3 = upper_value1
# y4 = lower_value1


# plot
# ax.fill_between(x2, y3, y4, alpha=.5, linewidth=0, color='r')
# ax.plot(x2, median1, linewidth=2, color='r')
# ax.set(xlim=(0, 10), xticks=np.arange(1, 59),
#        ylim=(0, 1), yticks=np.arange(0,1))


