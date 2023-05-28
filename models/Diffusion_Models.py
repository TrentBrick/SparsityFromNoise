import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import numpy as np
from models import BaseModel
import ipdb
from .TopK_Act import Top_K
#from .SDM_Base import SDMBase
from .TrackedMLPLayer import TrackedMLPLayer
from .InhibCircuit import InhibCircuit
from .TrackedActFunc import TrackedActFunc
from .model_utils import log_wandb #flatten_sequential
import torch.optim as optim 
import copy

def perc_closest_targets(pred, target):

    dists = torch.cdist(target.flatten(start_dim=1), pred, p=2.0)
    vals, inds = dists.min(dim=0) # what the logits are actually closest to. 
    # they should be closest to the original xs. 
    # this is only checking for values within the batch! 
    acc = (inds==torch.arange(len(target), device=pred.device)).type(torch.float).mean().detach()
    return acc 

class NoiseLayer(nn.Module):
    def __init__(self, noise_amount, noise_type="normal"):
        super().__init__()

        self.prev_noise_amount = copy.copy(noise_amount)
        self.noise_amount = noise_amount

        if noise_type == "normal":
            self.noise_sample = lambda x: self.noise_amount*torch.randn_like(x)

        else: 
            if self.noise_amount==0.0:
                self.noise_amount=0.00000000000001
            if noise_type == "laplace":
                self.noise_dist = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([self.noise_amount]))
                self.noise_sample = lambda x: self.noise_dist.sample(x.shape).squeeze().to(x.device).detach()
            elif noise_type == "expo":
                self.noise_dist = torch.distributions.exponential.Exponential(torch.tensor([self.noise_amount]))
                self.noise_sample = lambda x: self.noise_dist.sample(x.shape).squeeze().to(x.device).detach()
            elif noise_type == "poisson":
                self.noise_dist = torch.distributions.poisson.Poisson(torch.tensor([self.noise_amount]))
                self.noise_sample = lambda x: self.noise_dist.sample(x.shape).squeeze().to(x.device).detach()
            else: 
                raise NotImplementedError()

    def turn_off_diffusion_noise(self):
        #print("Turning off diffusion noise")
        self.prev_noise_amount = copy.copy(self.noise_amount)
        self.noise_amount = 0.0
        
    def turn_on_diffusion_noise(self):
        #print("Turning off diffusion noise")
        self.noise_amount = copy.copy(self.prev_noise_amount)

    def forward(self, x):
        return x + self.noise_sample(x)

class Diffusion_Base(BaseModel):
    def __init__(self, params):
        super().__init__(params)

        self.params = params
        self.nneurons = params.nneurons[0]
        #self.uses_tracked_mlp = True
        self.log_results =True 

        if params.classification:
            assert params.transpose==False, "Cant tranpose with classification"

        self.net_layers = []

        self.noise_layer = NoiseLayer(params.diffusion_noise, noise_type=params.noise_type)

        self.net_layers.append( self.noise_layer )

        self.noise_layer_ind = 1 if params.project_before_noise else 0

        if params.use_projection_matrix:
            insert_ind = 0 if params.project_before_noise else 1
        
            self.net_layers.insert(insert_ind, nn.Linear(params.input_size, params.input_size, bias=False))

        if self.params.adjust_diffusion: 
            self.diffusion_anneal_coef = -(self.params.diffusion_noise - self.params.terminal_diffusion_noise)/self.params.epochs_to_terminal_noise

    def setup_net_and_pointers(self):
        self.net = nn.Sequential( 
            *self.net_layers)
        #self.list_of_layers = flatten_sequential(self.net)

        # for now it is going to be the last layer before. need to skip over the last activation layer too.
        self.neuron_layer_ind = 2 if self.params.use_projection_matrix else 1
        
        self.X_a = lambda: self.net[self.neuron_layer_ind].layer
        self.X_a_activations = lambda:self.net[self.neuron_layer_ind+1].activation_summer

        #self.X_a_current_activations = lambda:self.net[self.neuron_layer_ind].curr_activations

        # need to go around the ReLU. 
        self.X_vT = lambda: self.net[self.neuron_layer_ind+2]
        # TODO: will this here stay up to date?? 
        self.X_v  = lambda: self.X_vT().weight.T 
        
        if self.params.transpose:
            del self.net[self.neuron_layer_ind+2].weight

    def adjust_diffusion_noise(self,):

        if self.trainer.current_epoch>self.params.epoch_to_start_noise_annealing:

            self.net[self.noise_layer_ind].noise_amount = np.clip( self.params.diffusion_noise + ( (self.trainer.current_epoch-self.params.epoch_to_start_noise_annealing) *self.diffusion_anneal_coef), 0.0, None )

        wandb.log({"Diffusion_Noise":self.net[self.noise_layer_ind].noise_amount})

    def loop_model_until(self, x, terminal_ind, start_ind=0):
        # noise, activations, 
        # +1 to ind so that it actually runs through this index too. 
        for i in range(start_ind, terminal_ind+1):
            x = self.net[i](x)
        return x

    def run_model_till_noise(self, x):
        return self.loop_model_until(x, self.noise_layer_ind)

    def run_after_noise(self, x_noise):
        # goes until the final outputs
        return self.loop_model_until(x_noise, len(self.net_layers)-1, start_ind=self.noise_layer_ind+1)

    def run_model_till_neuron_activations(self, x, apply_noise=True, apply_neuron_activation=True):
        start_ind = 0 if apply_noise else self.noise_layer_ind +1
        loop_incrementer = 1 if apply_neuron_activation else 0
        # this goes through the ReLU afterwards too. 
        return self.loop_model_until(x, self.neuron_layer_ind+loop_incrementer, start_ind=start_ind)

    def run_model_from_pre_neuron_act_func_to_end(self,z, pre_act_func=True):
        loop_incrementer = 0 if pre_act_func else 1
        return self.loop_model_until(z, len(self.net_layers)-1, start_ind=self.neuron_layer_ind+1+loop_incrementer)

    def return_noise_and_final_outputs(self, x):
        x_noise = self.run_model_till_noise(x)
        x = self.run_after_noise(x_noise)
        return x_noise, x 

    def log_cosine_sims(self):
        # log how close to transpose the hidden is to the output
        if not self.params.classification: 
            cos_sims = ( self.X_a().weight /torch.norm(self.X_a().weight , dim=1, keepdim=True)  *  self.X_vT().weight.T /torch.norm(self.X_vT().weight.T , dim=1, keepdim=True)  ).sum(dim=1)

            weighted_cos_sims = ( ( self.X_a_activations() / self.X_a_activations().sum() ) * cos_sims ).sum()

            log_wandb(self, {
                f"cosine_sim_to_transpose": cos_sims ,
                f"Activation_Weighted_cosine_sim_to_transpose": weighted_cos_sims ,
                f"Mean_cosine_sim_to_transpose": cos_sims.mean() ,
            })

    def log_closest_pattern_accuracy(self, og_x, x):
        if not self.params.classification and self.params.use_wandb: 
            noise_acc = perc_closest_targets(self.run_model_till_noise(og_x), og_x)
            recon_acc = perc_closest_targets(x, og_x)

            prefix = "TRAIN" if self.net.training else "VAL"
            
            log_wandb(self,{
                f"{prefix}_closest_to_target/perc_of_noise":noise_acc, 
                f"{prefix}_closest_to_target/perc_of_recon":recon_acc, 
            })


    def forward(self, x):
        # flatten x and do some other things. 

        if self.params.use_convmixer_transforms or self.params.normalize_n_transform_inputs:
            
            x = self.normalizer(x)

        if len(x)>2:
            x = x.flatten(start_dim=1)

        if self.params.transpose: 
            self.net[self.neuron_layer_ind+2].weight = self.X_a().weight.T

        og_x = x.clone()

        z = self.run_model_till_neuron_activations(x, apply_noise=True, apply_neuron_activation=True)
        x = self.run_model_from_pre_neuron_act_func_to_end(z, pre_act_func=False)
        
        self.log_closest_pattern_accuracy(og_x, x)

        return x, z


class FFN_DIFFUSION(Diffusion_Base):
    def __init__(self, params):
        super().__init__(params)

        if params.use_inhib_circuit is not False: 
            act_f = InhibCircuit(params, params.nneurons[0])
        else: 
            act_f = TrackedActFunc(params.act_func, params.nneurons[0], self.params.device, self.params.non_relu_act_threshold, layer_ind=0, use_wandb=params.use_wandb )

        self.net_layers += [
            TrackedMLPLayer(params.input_size, params.nneurons[0], params.device, use_bias=params.use_bias_hidden, layer_ind=0, use_shared_bias=params.use_shared_bias, use_wandb=params.use_wandb), 
                act_f
            ]

        for i in range(len(params.nneurons)-1):

            if params.use_inhib_circuit is not False: 
                act_f = InhibCircuit(params, params.nneurons[i+1])
            else: 
                act_f = TrackedActFunc(params.act_func, params.nneurons[i+1], self.params.device, self.params.non_relu_act_threshold,  layer_ind=i+1, use_wandb=params.use_wandb )

            self.net_layers += [
                TrackedMLPLayer(params.nneurons[i], params.nneurons[i+1], params.device, use_bias=params.use_bias_hidden, layer_ind=i+1, use_wandb=params.use_wandb),
                act_f
            ]
        
        # don't care about tracking the last layer. 
        self.net_layers.append( 
            nn.Linear(params.nneurons[-1], params.output_size, bias=params.use_bias_output) )

        self.setup_net_and_pointers()

    def forward(self, x, ):
        # will run through the whole model.
        return super().forward(x)


#########################

class SDM_DIFFUSION(Diffusion_Base):
    def __init__(self, params):
        # this will create the net with the noise and or projection layers first. 
        super().__init__(params)

        if params.k_approach=="LEARN_K_SHARED_BIAS":
            use_shared_bias = True
            a_func = nn.ReLU()
        else: 
            use_shared_bias = False
            a_func = Top_K(params, params.nneurons[0], module_ind=0)
            self.top_k = a_func
            
        activation_layer = TrackedActFunc( a_func , params.nneurons[0], self.params.device, self.params.non_relu_act_threshold, layer_ind=0, use_wandb=params.use_wandb )
        
        assert len(params.nneurons)==1, "havent implmented multi block SDM yet here!"
        # here is one full SDM module: 

        self.net_layers += [
            TrackedMLPLayer(params.input_size, params.nneurons[0], params.device, use_bias=params.use_bias_hidden, layer_ind=0, use_wandb=params.use_wandb, use_shared_bias=use_shared_bias, norm_weights=params.norm_addresses, norm_inputs=params.norm_addresses, positive_inputs=params.all_positive_weights),
            activation_layer,
            TrackedMLPLayer(params.nneurons[0], params.output_size, params.device, use_bias=params.use_bias_output, layer_ind=1, use_wandb=False, norm_weights=params.norm_values, norm_inputs=params.norm_activations)
        ]

        

        self.setup_net_and_pointers()

    #need to run each train step. 

    def forward(self, x, ):
        return super().forward(x)