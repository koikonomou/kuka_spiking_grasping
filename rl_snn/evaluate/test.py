import torch
import sys
import torch.nn as nn
sys.path.append('../')
from training.strain.networks import CriticNet
from training.snn_training.networks import ActorNetSpiking

model = training.snn_training.networks.ActorNetSpiking(14,5,1)

model.load_state_dict(torch.load("/home/katerina/snn_model/SNN_R1_snn_actor_model_1.pt"))
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])