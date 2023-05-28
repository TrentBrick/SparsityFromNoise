import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import numpy as np 

class TrackedActFunc(nn.Module):
    def __init__(self, act_func, nneurons, device, non_relu_act_threshold, layer_ind=0, use_wandb=True, wandb_prefix="" ):
        super().__init__()
        self.act_func = act_func
        self.use_wandb = use_wandb
        self.layer_ind = layer_ind 
        self.wandb_prefix = wandb_prefix

        if str(act_func) == "ReLU()" or str(act_func) == "Top_K()": 
            self.log_neuron_act_vals = False
            self.activity_threshold = 0.0
        else: 
            self.log_neuron_act_vals = True
            self.activity_threshold = non_relu_act_threshold

        print("activity threshold is:", self.activity_threshold)

        self.activation_summer = torch.zeros( nneurons ).to(device)

    def step_counter(self):
        self.act_func.step_counter()

    def wandb_push_dead_neurons(self):
        # log Neuron activations at end of epoch:
        # 
        if self.use_wandb:
        
            act_pdf = self.activation_summer/self.activation_summer.sum()

            z_inds = act_pdf == 0.0
            act_pdf[z_inds] += 1e-15
            neuron_ent = -(act_pdf * torch.log2(act_pdf)).sum()

            dic = {
                    f"layer_{self.layer_ind}/{self.wandb_prefix}fraction_dead_train_neurons": (
                        self.activation_summer < 0.00001
                    ).type(torch.float).mean(),

                    f"layer_{self.layer_ind}/{self.wandb_prefix}perc_time_neuron_active": act_pdf,

                    f"layer_{self.layer_ind}/{self.wandb_prefix}neuron_activity_pdf_entropy": neuron_ent,
                }

            wandb.log(
                    dic
                )
            
            self.activation_summer *=0

    def count_dead_neurons(self, neurons):
        if self.training:
            neurons = neurons.detach()
            if str(self.act_func) == "ReLU()": 
                # storing the actual neuron activity in this case
                self.activation_summer += neurons.sum(0)
            else: 
                self.activation_summer += (torch.abs(neurons)>self.activity_threshold).sum(0)

    def log_active_neurons(self, x):
        
        batched_mean_active_neurons = (torch.abs( x.detach()  )>self.activity_threshold).type(torch.float).mean(dim=1)

        wb_dict = {
            f"layer_{self.layer_ind}/{self.wandb_prefix}mean_Active_Neurons": batched_mean_active_neurons.mean(), 

            f"layer_{self.layer_ind}/{self.wandb_prefix}Active_Neurons": batched_mean_active_neurons
            }

        if self.log_neuron_act_vals:
            wb_dict[f"layer_{self.layer_ind}/{self.wandb_prefix}Neuron_Act_Vals_BInd_0"]= x[0].detach()
        
        if self.use_wandb:
            wandb.log(wb_dict)
        else: 
            if np.random.rand()>0.995:
                print("Mean active neurons for batch", wb_dict[f"layer_{self.layer_ind}/{self.wandb_prefix}mean_Active_Neurons"])

    def forward(self, x):
        x = self.act_func(x)
        self.log_active_neurons(x)
        if self.use_wandb:
            self.count_dead_neurons( x )
        return x
