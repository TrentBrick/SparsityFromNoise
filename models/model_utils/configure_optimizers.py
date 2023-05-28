import torch.optim as optim 

'''
When adding a new optimizer or LR, simply add a line to this dict with the relevant parameters. This is used to fill out `config_optimizers`
in the LightningModule body of `Base_model.py` and is abstracted away into this file for brevity and modularity. 
'''
def configure_optimizers_(self, verbose):
    lr = self.params.lr

    if "k_approach" in vars(self.params) and "LEARN_K" in self.params.k_approach:
        #dont add the learn_k parameter to the optimizer. want to add it separately. 
        params_to_opt = []
        for name, param in self.named_parameters():
            if param.requires_grad and  "learnt_k" not in name:
                    params_to_opt.append(param)
            else: 
                print("Not adding parameter", name )

    elif self.params.separate_bias_opts: 

        params_to_opt = []
        bias_to_opt = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue 
            if "bias" in name or len(param.shape) == 1:
                print("adding parameter to bias params:", name)
                bias_to_opt.append(param)
            else:
                params_to_opt.append(param)

    else: 
        # adding everything
        params_to_opt = list(self.parameters())

    ########### Set Optimizer ############
    if self.params.opt =="SGD": 
        optimizer = optim.SGD(params_to_opt,    lr=lr)
    elif self.params.opt == "SGDM":
        optimizer = optim.SGD(
            params_to_opt, lr=lr, momentum=self.params.sgdm_momentum)
    elif self.params.opt == "Adam": 
        optimizer = optim.Adam(params_to_opt, lr=lr,betas=self.params.adam_betas)
    elif self.params.opt ==  "RMSProp": 
        optimizer = optim.RMSprop(params_to_opt, lr=lr) 
    elif self.params.opt == "AdamW": 
        optimizer = optim.AdamW(params_to_opt, lr=lr, weight_decay=self.params.adamw_l2_loss_weight)
    elif self.params.opt == "AdaGrad": 
        optimizer = optim.Adagrad(params_to_opt, lr=lr)
    elif self.params.opt == "AdaFactor": 
        optimizer = add_opts.Adafactor(params_to_opt, lr=lr)
    elif self.params.opt == "Shampoo": 
        optimizer =add_opts.Shampoo(params_to_opt, lr=lr)

    elif self.params.opt == "SparseAdam": optimizer =SparseAdam(params_to_opt,
        lr=lr, betas=self.params.adam_betas)

    elif self.params.opt == "SparseSGDM": 
        optimizer =SparseSGDM(params_to_opt, lr=lr, momentum=self.params.sgdm_momentum)

    elif self.params.opt == "DemonAdam": 
        optimizer =DemonRanger(params_to_opt, 
                    lr=lr, 
                    #weight_decay=config.wd,
                    epochs = self.params.epochs_to_train_for,
                    #step_per_epoch = step_per_epoch, 
                    betas=(0.9,0.999,0.999), # restore default AdamW betas
                    nus=(1.0,1.0), # disables QHMomentum
                    k=0,  # disables lookahead
                    alpha=1.0, 
                    IA=False, # enables Iterate Averaging
                    rectify=False, # disables RAdam Recitification
                    AdaMod=False, #disables AdaMod
                    AdaMod_bias_correct=False, #disables AdaMod bias corretion (not used originally)
                    use_demon=True, #enables Decaying Momentum (DEMON)
                    use_gc=False, #disables gradient centralization
                    amsgrad=False # disables amsgrad
                    )
    else: 
        raise NotImplementedError("Need to implement optimizer")

    if "k_approach" in vars(self.params) and "LEARN_K" in self.params.k_approach:
        print("Learning k with a learning rate of:", self.params.lr*self.params.learn_k_lr_multiplier)
        k_lr = self.params.lr*self.params.learn_k_lr_multiplier
        if self.params.k_approach == "LEARN_K_SIGMOID":
            optimizer.add_param_group({'params': self.top_k.learnt_k, 'lr':k_lr})
        elif self.params.k_approach == "LEARN_K_REINFORCE":

            # IT IS JUST USING SGD FOR NOW. 

            reinforce_optimizer = optim.SGD(
            [self.top_k.learnt_k_suff_stats], lr=k_lr)

            optimizer = (optimizer, reinforce_optimizer)

        elif self.params.k_approach == "LEARN_K_SHARED_BIAS":
            pass 
        else: 
            raise NotImplementedError()

    elif self.params.separate_bias_opts: 
        #print(bias_to_opt)
        optimizer.add_param_group({'params': bias_to_opt, 'lr':self.params.separate_bias_opts_lr})
        print(optimizer)


    ########### Set LR SCHEDULER ############

    # TODO: need to be able to load in the learning rate scheduler. 
    if self.params.lr_scheduler == "StepLR": 
        self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer, self.params.lr_scheduler_step_size,
            gamma=self.params.step_lr_gamma, last_epoch=-1, verbose=False)
    elif self.params.lr_scheduler is None: 
        self.lr_scheduler = None 
    else: 
        raise NotImplementedError("Don't recognize LR scheduler")
    
    ########### Return to Base_model.py ############
    if verbose:
        print("length of net parameters", len(params_to_opt))

    return optimizer 