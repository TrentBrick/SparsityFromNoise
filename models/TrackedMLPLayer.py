import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb

'''class Custom_Sequential(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers

    def forward(self, x, log_results=True):

        for l in self.layers:
            x = l(x)

        return x'''

'''weight_sparsity_percentage = (
                torch.abs(self.X_a.weight.data) < self.params.sparsity_threshold
            ).sum(1) / (self.X_a.weight.data.shape[0]*self.X_a.weight.data.shape[1])

            wandb.log(
                {
                    "weight_sparsity_percentage": weight_sparsity_percentage,
                }
            )
'''


class TrackedMLPLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, device, use_bias, layer_ind, use_shared_bias=False, use_wandb=True, norm_weights=False, norm_inputs=False, positive_inputs=False ):
        super().__init__()
        self.tracked_mlp_layer = True 
        self.layer_ind = layer_ind
        self.use_bias = use_bias
        self.device = device
        self.use_shared_bias = use_shared_bias 
        self.use_wandb = use_wandb 
        self.norm_weights=norm_weights
        self.norm_inputs = norm_inputs
        self.positive_inputs= positive_inputs

        if self.use_shared_bias: 
            self.shared_bias = nn.Parameter(torch.zeros(1))
            self.use_bias = False # for the layer. 

        self.layer = nn.Linear(inp_dim, out_dim, bias=self.use_bias)
        # pointer to the model weights. 
        self.weight = self.layer.weight
        self.bias = self.layer.bias 

        

    def enforce_l2_norm_weights(self):
        # OVERWRITING THE BASE MODEL HERE.
        # L2 norm all the neural network weights:
        if self.norm_weights:
            with torch.no_grad():
                self.weight.data /= torch.norm(self.weight.data, dim=1, keepdim=True)

    def log_weight_info(self):
        if self.use_wandb:
            l2_norm = torch.norm(self.layer.weight, dim=1)
            wandb.log( {
                f"layer_{self.layer_ind}/Mean_L2_norm": l2_norm.mean(),

                #f"layer_{self.layer_ind}/Activation_Weighted_L2_norm": ( ( self.activation_summer / self.activation_summer.sum() ) * l2_norm ).sum(),

                f"layer_{self.layer_ind}/L2_norm": l2_norm,
            })


    def log_bias_params(self, x):

        bias_vals = self.shared_bias if self.use_shared_bias else self.layer.bias 

        with torch.no_grad():
            pre_bias = (self.layer.weight@x.T).T
            post_bias = pre_bias + bias_vals

            pre_pos = (pre_bias>0)
            post_pos = (post_bias>0)
            neg_to_pos = torch.logical_and(~pre_pos, post_pos)
            pos_to_neg = torch.logical_and(pre_pos, ~post_pos)

        if self.use_wandb:
            wandb.log( {
                f"layer_{self.layer_ind}/Mean_neuron_bias_terms": bias_vals.detach().mean(), 

                f"layer_{self.layer_ind}/frac_neg_bias_terms": (bias_vals<0.0).type(torch.float).detach().mean(), 

                f"layer_{self.layer_ind}/neuron_bias_terms": bias_vals.detach(),

                f"layer_{self.layer_ind}/Mean_pre_bias_Active_neurons": pre_pos.type(torch.float).mean(),

                f"layer_{self.layer_ind}/Mean_neg_to_pos_neurons": neg_to_pos.type(torch.float).mean(),

                f"layer_{self.layer_ind}/Mean_pos_to_neg_neurons": pos_to_neg.type(torch.float).mean(),

                })

    def forward(self, x):

        if self.positive_inputs: 
            x = F.relu(x) 

        # l2 norm the input data
        if self.norm_inputs:
            x = x / torch.norm(x, dim=1, keepdim=True)
 
        # eg if convergence_iter=0
        # this is set by the parent and given as a function that can be called. 
        out_x = self.layer(x)

        if self.use_shared_bias:
            out_x += self.shared_bias

        if self.use_wandb:
            self.log_weight_info()

            if self.use_bias or self.use_shared_bias:
                self.log_bias_params(x)

        # need to pass it through to the higher up layers. 
        return out_x