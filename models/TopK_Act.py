import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import functional as F
import wandb 

class Top_K(nn.Module):
    def __init__(self, params, nneurons, module_ind=0):
        super().__init__()
        self.params = params
        self.module_ind = module_ind
        self.k_max = nneurons
        self.nneurons = nneurons
        self.k_dim=1
        self.return_cos_thresh = False 

        if self.params.top_k_blocks >1:
            self.k_dim=2

        assert self.nneurons%self.params.top_k_blocks==0, "Need to have number of blocks divide into the number of neurons"
        self.neurons_per_block = self.nneurons//self.params.top_k_blocks
        assert self.params.k_min % self.params.top_k_blocks==0, "k_min should be divisible by the number of blocks"

        if self.params.k_approach == "LEARN_K_SIGMOID":
            self.learnt_k = nn.Parameter(  (self.params.learn_k_init*torch.ones(1))+0.1,requires_grad=True)

            self.window_perc = params.learn_k_window_perc
            self.window_num = int(nneurons*self.window_perc)
        elif self.params.k_approach == "LEARN_K_REINFORCE":
            mut1, mut2 = 6, 1
            self.learnt_k_suff_stats = nn.Parameter( torch.cat( [mut1*torch.ones(1, device=params.device), mut2*torch.ones(1, device=params.device) ] ),requires_grad=True)
            
            self.learnt_k_pdf = torch.distributions.normal.Normal( self.learnt_k_suff_stats[0], self.learnt_k_suff_stats[1])

        # track activations for GABA
        elif "GABA_SWITCH" in params.k_approach:
            # using a buffer. 
            self.register_buffer("neuron_activation_counters", torch.zeros((1,self.nneurons) , requires_grad=False) )

            if self.params.k_approach == "GABA_SWITCH_ACT_BIN":
                self.linear_coef_threshold = self.params.num_binary_activations_for_gaba_switch
                self.neuron_counter_name = "neuron_binary_activation_counters"
            elif self.params.k_approach == "GABA_SWITCH_ACT":
                self.linear_coef_threshold = self.params.num_activations_for_gaba_switch
                self.neuron_counter_name = "neuron_activation_counters"
            elif self.params.k_approach == "GABA_SWITCH_GRAD": 
                self.linear_coef_threshold = self.params.num_grads_for_gaba_switch
                self.neuron_counter_name = "neuron_gradient_counters"
            else: 
                raise Exception("This is a gaba switch but it is not defined what type")
            
        elif "LINEAR_DECAY" in self.params.k_approach or "EXP_DECAY" in self.params.k_approach:
            self.register_buffer("curr_step", torch.zeros((1) , requires_grad=False) )

            if "LINEAR_DECAY" in self.params.k_approach:
                self.linear_coef = (
                    -(self.k_max - self.params.k_min) / self.params.k_transition_epochs
                )

            if "EXP_DECAY" in self.params.k_approach:
                self.exp_decay_coef = (self.params.k_min/self.k_max)**(1/self.params.k_transition_epochs)

    def step_counter(self):
        self.curr_step+=1

    def get_curr_k(self):
        # linearly (across epochs_to_train_for) reduce k value between k_max and k_min
        # returns an int. 
        # torch.topk expects an int. Need to bring curr_step to cpu to ultimately convert it into an int. It is on the GPU because it is a registered buffer. 
        if "LINEAR_DECAY" in self.params.k_approach: 
            k = self.k_max + (self.linear_coef * self.curr_step.cpu())
        elif "EXP_DECAY" in self.params.k_approach:
            k = self.k_max*(self.exp_decay_coef**self.curr_step.cpu())

        else:
            k = self.params.k_min

        #k = torch.clip( k, min=self.params.k_min, max=self.k_max)

        # min then max. 
        k = int(np.clip( k, self.params.k_min, self.k_max))

        if "GABA_SWITCH" in self.params.k_approach:
            k=k+1 # to subtract by this value

        # +1 because we are finding the value for this lowest one to then subtract from. 
        return k

    def compute_learn_k_window(self, k_int):
        # clamp these values within the min and max range. 
        max_window = np.minimum( k_int+(self.window_num//2), self.nneurons-1)
        min_window = np.maximum( k_int-(self.window_num//2), 0 )
        # update window num with the actual amount allowed. 
        allowed_window_num = max_window-min_window
        num_below_k, num_above_k = k_int-min_window, max_window-k_int 

        return max_window, allowed_window_num, num_below_k, num_above_k

    def give_learn_k_weights(self, top_k_val, k_int, allowed_window_num, num_below_k, num_above_k, bs):
        weight_vals = torch.ones(top_k_val, device=self.params.device)#_like(inds).type(torch.float)
        # need to modify this with the sigmoid weights. 
        # squeezing the whole function into this window.
        lin_lower = -num_below_k*(20/allowed_window_num)
        lin_upper = num_above_k*(20/allowed_window_num) 
        sigmoid_span = torch.linspace(lin_lower, lin_upper, allowed_window_num, device=self.params.device)
        k_decimal_gap = self.learnt_k - k_int
        sigmoid_weights = -F.sigmoid(sigmoid_span-k_decimal_gap) +1
        weight_vals[-allowed_window_num:] *= sigmoid_weights
        weight_vals = weight_vals.unsqueeze(0).expand(bs, -1)

        return weight_vals 

    def sample_from_reinforce_dist(self, bs):
        k_samples = self.learnt_k_suff_stats[0] + self.learnt_k_suff_stats[1]*torch.randn((bs, 1), device=self.params.device)

        # clamp to the max value: 
        k_samples = torch.clamp(k_samples, max=np.log(self.nneurons), min=np.log(1)  )

        self.k_log_probs = self.learnt_k_pdf.log_prob(k_samples).squeeze()

        # assuming that it is in log space!!! 
        k_samples = torch.exp(k_samples.squeeze())

        # all of these are different. 
        k_ints = torch.round(k_samples).cpu().detach().numpy().astype(int)
        return k_ints 

    def forward(self, x):
        
        x = F.relu(x)
        
        # for learning k need to get the larger value and then decide if to reduce it or not
        if self.params.k_approach == "LEARN_K_SIGMOID":
            k_int = int(torch.round(self.learnt_k).cpu().detach().numpy())
            # here curr k is the maximum value of the window!
            max_window, allowed_window_num, num_below_k, num_above_k = self.compute_learn_k_window(k_int)
            top_k_val = max_window # to use in the topK operation. 
            sort_k = True
        elif self.params.k_approach == "LEARN_K_REINFORCE":
            k_ints = self.sample_from_reinforce_dist(len(x))
            top_k_val = max(k_ints)
            sort_k = True
        else: 
            top_k_val = self.get_curr_k()
            sort_k = False

        if self.params.top_k_blocks >1:
            assert "LEARN_K" not in self.params.k_approach, "Not implemented or debugged yet!"
            x = x.view( x.shape[0], x.shape[1], self.params.top_k_blocks, self.neurons_per_block )
            top_k_val = top_k_val // self.params.top_k_blocks

        vals, inds = torch.topk(x, top_k_val, dim=self.k_dim, sorted=sort_k)
        inhib_amount, _ = torch.min(vals.detach(), dim=self.k_dim, keepdim=True)

        if "GABA_SWITCH" in self.params.k_approach:
            # get threshold coefficient
            gaba_response = torch.clamp(
                -1 + ( (2 / self.linear_coef_threshold) * self.neuron_activation_counters), min=-1, max =1
            ).type_as(x)

            if self.params.use_wandb:
                wandb.log( {
                        f"TopK_Act_{self.module_ind}/{self.neuron_counter_name}": self.neuron_activation_counters,
                        f"TopK_Act_{self.module_ind}/gaba_response": gaba_response
                    })
        else:
            gaba_response = 1

        # apply inhibition
        if "MASK" in self.params.k_approach:
            top_k_mask = torch.zeros_like(x)
            x = x * top_k_mask.scatter(self.k_dim, inds, 1) 

        elif self.params.k_approach == "LEARN_K_SIGMOID":
            # apply the decimal weighting to the last index of each. does this independently of the k approach being taken. 
            weight_vals = self.give_learn_k_weights(top_k_val, k_int, allowed_window_num, num_below_k, num_above_k, len(x))

            top_k_mask = torch.zeros_like(x)
            x = x * top_k_mask.scatter(self.k_dim, inds, weight_vals)

        elif self.params.k_approach == "LEARN_K_REINFORCE":

            top_k_mask = torch.zeros_like(x)
            # repeat the first one the number of times for the gap. 
            
            mask_inds = torch.stack( [ torch.cat([inds[i, :ak], inds[i,0].repeat( top_k_val - ak)]) for i, ak in enumerate(k_ints)] )
            x = x * top_k_mask.scatter(self.k_dim, mask_inds, 1)

        else: 
            # GABA switch or Subtracts (including learning k through a shared bias term!!)
            x = F.relu(x - (gaba_response * inhib_amount))

        if self.params.top_k_blocks >1:
            x = x.flatten(start_dim=2)
            
        if "GABA_SWITCH_ACT" in self.params.k_approach:
            if self.params.k_approach == "GABA_SWITCH_ACT_BIN":
                self.neuron_activation_counters += torch.sum(x>0, dim=0, keepdim=True)
            elif self.params.k_approach == "GABA_SWITCH_ACT":
                self.neuron_activation_counters += torch.sum(x.detach(), dim=0, keepdim=True)

        if self.params.use_wandb:
            
            log_dict = {
                f"TopK_Act_{self.module_ind}/mean_inhib_amount": inhib_amount.mean(),

                f"TopK_Act_{self.module_ind}/post_TopK_active_number": (F.relu(x.detach())>0.0).type(torch.float).mean(),
            }

            if self.params.k_approach == "LEARN_K_SIGMOID":
                log_dict[f"TopK_Act_{self.module_ind}/k"]= self.learnt_k.item()
            elif self.params.k_approach == "LEARN_K_REINFORCE":
                log_dict[f"TopK_Act_{self.module_ind}/mean"]=self.learnt_k_suff_stats[0].exp().item()
                log_dict[f"TopK_Act_{self.module_ind}/variance"]=self.learnt_k_suff_stats[1].exp().item()
                log_dict[f"TopK_Act_{self.module_ind}/mean_k_used"]= k_ints.mean()
            else: 
                log_dict[f"TopK_Act_{self.module_ind}/k"]= top_k_val

            

            wandb.log(log_dict)

        if self.return_cos_thresh: 
            return x, inhib_amount

        return x 