import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib

def plot_results(data):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    run_num = len(data["final_state"])
    time = []
    color_list = []
    for r in range(run_num):
        print(data["time"][r])

        if data["final_state"][r] == 1:
            color_list.append('Success%r'%r)
            time.append(data["time"][r])
        if data["final_state"][r] == 2:
            color_list.append("Failure")
            time.append(data["time"][r])
        if data["final_state"][r] == 3:
            color_list.append("Time out")
            time.append(data["time"][r])
        # matplotlib.pyplot.plot(state_list,path_time[0:3])
    # error = np.random.rand(len(color_list))
    ax.barh(color_list, time,  align='edge')
    ax.set_yticks(color_list, labels=color_list)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time')
    ax.set_title('Success Rate')
    plt.show()
def analyze_run(data):
    """
    Analyze success rate, path distance, path time, and path avg spd
    :param data: run_data
    :return: state_list, path_dis, path_time, path_spd
    """
    run_num = len(data["final_state"])
    state_list = [0, 0, 0, 0, 0]
    path_dis = np.zeros(run_num)
    path_time = np.zeros(run_num)
    path_spd = np.zeros(run_num)
    for r in range(run_num):
        if data["final_state"][r] == 1:
            state_list[0] += 1
            tmp_overll_path_dis = 0
            for d in range(len(data["path"][r]) - 1):
                rob_pos = data["path"][r][d]
                next_rob_pos = data["path"][r][d + 1]
                tmp_dis = math.sqrt((next_rob_pos[0] - rob_pos[0]) ** 2 + (next_rob_pos[1] - rob_pos[1]) ** 2)
                tmp_overll_path_dis += tmp_dis
            path_dis[r] = tmp_overll_path_dis
            path_time[r] = data["time"][r]
            path_spd[r] = path_dis[r] / path_time[r]
        elif data["final_state"][r] == 2:
            state_list[1] += 1
        elif data["final_state"][r] == 3:
            state_list[2] += 1
        else:
            print("FINAL STATE TYPE ERROR ...")
        # print(state_list, path_dis, path_time, path_spd)
    return state_list, path_dis, path_time, path_spd


def plot_robot_paths(data, goal_list, env_size=((-10, 10), (-10, 10))):
    """
    Plot Robot Path from experiment
    :param path: robot path
    :param final_state: final states
    :param poly_list: obstacle poly list
    """
    path = data["path"]
    final_state = data["final_state"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    for i, p in enumerate(path, 0):
        p_x = [p[num][0] for num in range(len(p))]
        p_y = [p[num][1] for num in range(len(p))]
        if final_state[i] == 1:
            ax[0].plot(p_x, p_y, color='#4169E1', linestyle='-', lw=0.5)
            ax[0].plot([p_x[0]], [p_y[0]], 'bo')
            ax[0].plot([p_x[-1]], [p_y[-1]], 'ro')
        elif final_state[i] == 2:
            ax[1].plot(p_x, p_y, color='#4169E1', linestyle='-', lw=0.8)
            ax[1].plot([p_x[0]], [p_y[0]], 'bo')
            ax[1].plot([p_x[-1]], [p_y[-1]], 'rx')
            ax[1].plot([goal_list[i][0]], [goal_list[i][1]], 'ro')
            ax[1].plot([p_x[-1], goal_list[i][0]], [p_y[-1], goal_list[i][1]], 'r--', lw=0.8)
        elif final_state[i] == 3:
            ax[1].plot(p_x, p_y, color='#4169E1', linestyle='-', lw=0.8)
            ax[1].plot([p_x[0]], [p_y[0]], 'bo')
            ax[1].plot([p_x[-1]], [p_y[-1]], 'go')
            ax[1].plot([goal_list[i][0]], [goal_list[i][1]], 'ro')
            ax[1].plot([p_x[-1], goal_list[i][0]], [p_y[-1], goal_list[i][1]], 'r--', lw=0.8)
        else:
            print("Wrong Final State Value ...")
    ax[0].set_xlim(env_size[0])
    ax[0].set_ylim(env_size[1])
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title("Success Routes")
    ax[1].set_xlim(env_size[0])
    ax[1].set_ylim(env_size[1])
    ax[1].set_aspect('equal', 'box')
    ax[1].set_title("Failure Routes (Collision + Overtime)")
