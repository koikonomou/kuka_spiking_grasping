import torch
import torch.nn as nn
import numpy as np
import random
import pickle
import math
import sys
sys.path.append('../')
# from training.train_ddpg.ddpg_networks import ActorNet
from training.snn_training.agent import AgentSpiking
from training.snn_training.networks import ActorNetSpiking

def load_test_actor_network(dir, state_num=22, action_num=2, dim=(256, 256, 256)):
    """
    Load actor network for testing
    :param dir: directory of pt file
    :return: actor_net
    """
    actor_net = ActorNet(state_num, action_num,
                         hidden1=dim[0],
                         hidden2=dim[1],
                         hidden3=dim[2])
    actor_net.load_state_dict(torch.load(dir, map_location=lambda storage, loc: storage))
    return actor_net


def load_test_actor_snn_network(weight_dir, bias_dir, device, batch_window=50,
                                state_num=14, action_num=5, dim=(256, 256, 256)):
    """
    Load actor snn network for testing
    :param weight_dir: directory of numpy weights
    :param bias_dir: directory of numpy bias
    :param state_num: number of states
    :param action_num: number of actions
    :param dim: net dim
    :return: actor_net
    """
    actor_net = ActorNetSpiking(state_num, action_num, device,
                                batch_window=batch_window,
                                hidden1=dim[0],
                                hidden2=dim[1],
                                hidden3=dim[2])
    weights = pickle.load(open(weight_dir, 'rb'))
    bias = pickle.load(open(bias_dir, 'rb'))
    actor_net.fc1.weight = nn.Parameter(torch.from_numpy(weights[0]))
    actor_net.fc2.weight = nn.Parameter(torch.from_numpy(weights[1]))
    actor_net.fc3.weight = nn.Parameter(torch.from_numpy(weights[2]))
    actor_net.fc4.weight = nn.Parameter(torch.from_numpy(weights[3]))
    actor_net.fc1.bias = nn.Parameter(torch.from_numpy(bias[0]))
    actor_net.fc2.bias = nn.Parameter(torch.from_numpy(bias[1]))
    actor_net.fc3.bias = nn.Parameter(torch.from_numpy(bias[2]))
    actor_net.fc4.bias = nn.Parameter(torch.from_numpy(bias[3]))
    return actor_net



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
