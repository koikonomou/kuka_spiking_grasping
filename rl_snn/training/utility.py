import math
import random
import numpy as np
from shapely.geometry import Point, Polygon




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


def snn_state_2_spike_value_state(state, spike_state_num):
    """
    Transform snn state to Spike Value State for SNN
    :param state: single snn state
    :param spike_state_num: number of spike states
    :param goal_dir_range: range of goal dir
    :return: spike_state_value
    """
    spike_state_value = [0 for _ in range(spike_state_num)]
    spike_state_value[0] = state[0]
    spike_state_value[1:7] = state[1]
    spike_state_value[7:13] = state[2]
    spike_state_value[13] = state[3]
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
    rescale_state_value[2] = state[1]
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
    if state[0] > 0:
        spike_state_value[0] = state[0]
        spike_state_value[1] = 0
    else:
        spike_state_value[0] = 0
        spike_state_value[1] = state[0]
    spike_state_value[2:8] = state[1]
    spike_state_value[8:14] = state[2]
    spike_state_value[14] = state[3]
    # print(spike_state_value)
    # print ("spike_state_value", len(spike_state_value))
    return spike_state_value
