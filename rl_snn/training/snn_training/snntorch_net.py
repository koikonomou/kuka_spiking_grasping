import torch
import numpy as np
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


alpha = 0.9
beta = 0.85

NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5





class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        # print('rect', grad_input * spike_pseudo_grad.float() )
        return grad_input * spike_pseudo_grad.float()


# Initialize surrogate gradient
spike_grad1 = surrogate.fast_sigmoid()  # passes default parameters from a closure
# spike_grad2 = surrogate.FastSigmoid.apply  # passes default parameters, equivalent to above
# spike_grad3 = surrogate.fast_sigmoid(slope=50)  # custom parameters from a closure

# Define Network
class ActorNetSpiking(nn.Module):
    def __init__(self, state_num, action_num, device, batch_window=50, hidden1=256, hidden2=256, hidden3=256):


        super(ActorNetSpiking, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        # self.spike_grad1 = PseudoSpikeRect.apply
     # Initialize layers, specify the ``spike_grad`` argument
        self.fc1 = nn.Linear(self.state_num, self.hidden1)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad1)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad1)
        self.fc4 = nn.Linear(self.hidden3, self.action_num)
        self.lif4 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad1)

    def forward(self, x, batch_size):

        spk1 = torch.zeros(batch_size, self.hidden1, device=self.device)
        syn1 = torch.zeros(batch_size, self.hidden1, device=self.device)
        mem1 = torch.zeros(batch_size, self.hidden1, device=self.device)
        spk2 = torch.zeros(batch_size, self.hidden2, device=self.device)
        syn2 = torch.zeros(batch_size, self.hidden2, device=self.device)
        mem2 = torch.zeros(batch_size, self.hidden2, device=self.device)
        spk3 = torch.zeros(batch_size, self.hidden3, device=self.device)
        syn3 = torch.zeros(batch_size, self.hidden3, device=self.device)
        mem3 = torch.zeros(batch_size, self.hidden3, device=self.device)
        spk4 = torch.zeros(batch_size, self.action_num, device=self.device)
        syn4 = torch.zeros(batch_size, self.action_num, device=self.device)
        mem4 = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)

        for step in range(self.batch_window):
            input_spike = x[:, :, step]
            cur1 = self.fc1(input_spike)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            cur3 = self.fc3(spk2)
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            cur4 = self.fc4(spk3)
            spk4, syn4, mem4 = self.lif4(cur4, syn4, mem4)
            fc4_sumspike += spk4
        out = fc4_sumspike / self.batch_window
        return out