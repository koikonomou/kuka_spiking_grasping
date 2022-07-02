import math
import random
import numpy as np
from shapely.geometry import Point, Polygon



def gen_rand_list_env(size,robot_goal_diff=1.0):
    # robot_init_pose = [0]*35
    # print (robot_init_pose)
    robot_init_pose = [ 4.62, -5.35, 0.8, 7.68, 2.17, -1.45, 1.0, 
                        7.68, -9.16, 1.13, -7.53, -6.94, 4.32, 1,
                        -4.16, 4.61, 1.12, -5.75, 0.01, 4.32, 0.99,
                        0.28, 2.51, 1.12, -6.24, 1.014, 4.32, 0.99,
                        0.60, 5.20, 1.13, 8.80, 0.01, 3.07, 0.99]

    return robot_init_pose


def network_2_robot_action_decoder(action, j1_max, j1_min, j2_max, j2_min, j3_max, j3_min, j4_max, j4_min, j5_max, j5_min):

    """
    Decode action from network output to action for the robot
    :param action: action from network output
    :param ji_max: position max in rad
    :param ji_min: negative min value in rad
    :return: robot_action
    """

    joint_a1 = action[0] * (j1_max - j1_min) + j1_min
    joint_a2 = action[1] * (j2_max - j2_min) + j2_min
    joint_a3 = action[2] * (j3_max - j3_min) + j3_min
    joint_a4 = action[3] * (j4_max - j4_min) + j4_min
    joint_a5 = action[4] * (j5_max - j5_min) + j5_min

    return [joint_a1, joint_a2, joint_a3, joint_a4, joint_a5]


def snn_state_2_spike_value_state(state, spike_state_num,
                                   goal_dir_range=math.pi):
    """
    Transform snn state to Spike Value State for SNN
    :param state: single snn state
    :param spike_state_num: number of spike states
    :param goal_dir_range: range of goal dir
    :return: spike_state_value
    """
    # print("state", len(state))
    spike_state_value = [0 for _ in range(spike_state_num)]
    if state[0][0] > 0:
        spike_state_value[0] = state[0][0] / goal_dir_range
        spike_state_value[1] = 0
    else:
        spike_state_value[0] = 0
        spike_state_value[1] = abs(state[0][0]) / goal_dir_range
    spike_state_value[2:37] = state[1]
    spike_state_value[37:67] = state[2]
    spike_state_value[67] = state[3][0]
    # print ("spike_state_value", len(spike_state_value))
    return spike_state_value

def ddpg_state_rescale(state, spike_state_num,
                                   goal_dir_range=math.pi):
    """
    Transform snn state to Spike Value State for SNN
    :param state: single snn state
    :param spike_state_num: number of spike states
    :param goal_dir_range: range of goal dir
    :return: spike_state_value
    """
    # print("state", len(state))
    rescale_state_value = [0 for _ in range(spike_state_num)]
    if state[0][0] > 0:
        rescale_state_value[0] = state[0][0] / goal_dir_range
        rescale_state_value[1] = 0
    else:
        rescale_state_value[0] = 0
        rescale_state_value[1] = abs(state[0][0]) / goal_dir_range
    rescale_state_value[2:37] = state[1]
    rescale_state_value[37:67] = state[2]
    rescale_state_value[67] = state[3][0]
    # print ("rescale_state_value", len(rescale_state_value))
    return rescale_state_value

def ddpg_state_2_spike_value_state(state, spike_state_num,
                                   goal_dir_range=math.pi):
    """
    Transform snn state to Spike Value State for SNN
    :param state: single snn state
    :param spike_state_num: number of spike states
    :param goal_dir_range: range of goal dir
    :return: spike_state_value
    """
    # print("state", spike_state_num)
    # print(state)
    spike_state_value = [0 for _ in range(spike_state_num)]
    if state[0][0] > 0:
        spike_state_value[0] = state[0][0] / goal_dir_range
        spike_state_value[1] = 0
    else:
        spike_state_value[0] = 0
        spike_state_value[1] = abs(state[0][0]) / goal_dir_range
    spike_state_value[2:37] = state[1]
    spike_state_value[37:67] = state[2]
    spike_state_value[67] = state[3][0]
    # print ("spike_state_value", len(spike_state_value))
    return spike_state_value
