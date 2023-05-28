import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from .TrackedActFunc import TrackedActFunc

class InhibCircuit(nn.Module):
    def __init__(self, params, nneurons):
        super().__init__()

        """
        use_inhib_circuit = False #"ALL-TO-ALL", #"INTERNEURON"
        only_inhibitory = False, # will apply ReLU activation and do a subtraction operation
        num_inhib_circuit_steps = 1,
        """

        self.use_inhib_circuit = params.use_inhib_circuit
        self.num_inhib_circuit_steps = params.num_inhib_circuit_steps

        self.inhib_act_func = nn.ReLU() if params.only_inhibitory else nn.Identity()
        self.update_step_size = params.inhib_circuit_step_size

        if self.use_inhib_circuit == "ALL-TO-ALL":
            self.inhib_circuit = nn.Linear(nneurons, nneurons, bias=False)

        elif self.use_inhib_circuit == "INTERNEURON":
            self.inhib_circuit = nn.Sequential(*[
                nn.Linear(nneurons, 1, bias=True),
                nn.ReLU(),
                nn.Linear(1,nneurons, bias=False)
                ])

        else: 
            raise NotImplementedError()

        self.start_tracked_ReLU = TrackedActFunc(nn.ReLU(), params.nneurons[0],  params.device, params.non_relu_act_threshold, layer_ind=0,use_wandb=params.use_wandb, wandb_prefix = "PreInhib_Circuit_" )
        self.end_tracked_ReLU = TrackedActFunc(nn.ReLU(), params.nneurons[0], params.device, params.non_relu_act_threshold, layer_ind=0, use_wandb=params.use_wandb )

        self.activation_summer = self.end_tracked_ReLU.activation_summer

    def wandb_push_dead_neurons(self):
        self.start_tracked_ReLU.wandb_push_dead_neurons()
        self.end_tracked_ReLU.wandb_push_dead_neurons()

    def forward(self, og_mu):
        # takes in the mu code from before. 
        # thus we plug it in before the ReLU activation function. 
        #import ipdb 
        #ipdb.set_trace()
        # this is just to track how things are coming in.
        mu = torch.clone(og_mu)
        _ = self.start_tracked_ReLU(mu.detach())

        # should bias term only be applied at the z step rather than within the delta step? . so then it is not affected by the 

        for i in range(self.num_inhib_circuit_steps):

            z = F.relu(mu)
            new_mu = og_mu - self.inhib_act_func(self.inhib_circuit(z))
            delta_mu = new_mu - mu 
            mu = (self.update_step_size * delta_mu) + mu
            
        z = self.end_tracked_ReLU(mu)

        return z