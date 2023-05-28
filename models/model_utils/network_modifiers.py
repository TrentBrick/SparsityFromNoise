import torch 
import torch.nn.utils.prune as prune

'''def flatten_sequential(net):
    list_of_layers = []
    for m in net: 
        if isinstance(m, torch.nn.modules.container.Sequential):
            for l in m:  
                list_of_layers.append(l)
        else: 
            list_of_layers.append(m)
    return list_of_layers'''

def enforce_positive_weights(model_params, can_enforce_positive_weights_on_bias_terms):
    # WARNING: this means the interneuron bias terms (and any others I introduce later) cant be negative removing a firing threshold for the interneuron.

    for p in list(model_params):
        # check if p is a bias term or global threshold.
        if len(p.shape) == 1 and not can_enforce_positive_weights_on_bias_terms:
            continue
            # otherwise wont do anything here to these neurons. 
        p.data.clamp_(0)

def push_train_batch_end_commands_to_network(self):

    # assumes that need to be using SDM regime to operate any of these. 
    if self.params.use_sdm: 

        if self.params.all_positive_weights:
            # iterates through and applies to all layers. 
            enforce_positive_weights(self.parameters(), self.params.can_enforce_positive_weights_on_bias_terms)

        # some sort of normalization is happening somewhere!
        if self.params.norm_addresses or self.params.norm_values:
            # always use .net to refer to the sequential model. 
            for module_name, module in self._modules.items():

                if isinstance(module, torch.nn.modules.container.Sequential):
            
                    for layer in module:
                        if getattr(layer, 'enforce_l2_norm_weights', False):
                            layer.enforce_l2_norm_weights()

def push_epoch_end_commands_to_network(self):

    # this happens at the top class level. 
    if getattr(self.params, 'adjust_diffusion', False):
        self.adjust_diffusion_noise()

    # try to store the cosine similarities between the address and value vectors this is for the model as a whole. 
    if getattr(self, 'log_cosine_sims', False):
        self.log_cosine_sims()

    if "net" in self._modules.keys():  
        for sub_module in self._modules['net']:

            if getattr(sub_module, 'wandb_push_dead_neurons', False):
                sub_module.wandb_push_dead_neurons()

            if "k_approach" in self.params and "DECAY" in self.params.k_approach and getattr(sub_module, 'step_counter', False):
                sub_module.step_counter()

"""
        # updated the L1 loss weight. 
        if self.params.l1_loss_weight is not None and self.params.l1_growth_rate is not None:
            self.params.l1_loss_weight = np.minimum(self.params.l1_growth_rate+self.params.l1_loss_weight,
                                                            self.params.l1_max_val)
            log_wandb(self, {'l1_loss_weight': self.params.l1_loss_weight})
        
"""




def prune_granules(params, net):
    # TODO: implement for Microzones

    num_dends_on = int(
        params.input_size* (1-params.granule_sparsity_percentage))

    if params.prune_smallest:
        # needs to happen not at init but at the start of training!
        _ , dendrite_inds = torch.topk(net.weight.data, num_dends_on, dim=1, sorted=False )
    else: 
        # prune randomly
        dendrite_inds = [np.random.choice(np.arange(params.input_size),
                                                   size=num_dends_on
    , replace=False) for _ in range(params.nneurons[0])]

        dendrite_inds = torch.Tensor(dendrite_inds).to( params.device)

    granule_prune_mask = torch.zeros_like(net.weight)
    
    granule_prune_mask = granule_prune_mask.scatter(
        1, dendrite_inds.type(torch.long), 1)
    
    return prune.custom_from_mask(
        net, "weight", granule_prune_mask)