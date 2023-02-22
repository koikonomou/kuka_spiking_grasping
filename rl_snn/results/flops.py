from flopth import flopth
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from training.snn_training.networks import ActorNetSpiking

from training.train_ddpg.ddpg_networks import ActorNet, CriticNet
import torch
import torch.nn as nn


actor_net_dim=[14, 256, 256]
path ='/home/katerina/ddpg_model/2022_09_22-12_47_37_PM_ddpg_actor_model_4.pt'
actor_net = ActorNet(14, 5,
                          hidden1=actor_net_dim[0],
                          hidden2=actor_net_dim[1],
                          hidden3=actor_net_dim[2])
actor_net = torch.load(path)
sum_flops = flopth(actor_net, in_size=(1,68))
print(sum_flops)