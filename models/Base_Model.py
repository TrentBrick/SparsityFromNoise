import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import wandb
from .model_utils import *
import ipdb
import torchvision
import torchvision.transforms as transforms



######## BASE MODEL FOR FUNCTIONS THAT ARE USED ACROSS DIFFERENT NETWORK ARCHITECTURES #######

class BaseModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.automatic_optimization = False # means I need to call gradient step myself. 
        self.params = params
        self.save_hyperparameters({"empty":None})

        if self.params.use_convmixer_transforms or self.params.normalize_n_transform_inputs:
            if "CIFAR" not in params.dataset_name:
                raise NotImplementedError()

            self.normalizer = transforms.Normalize(params.cifar10_mean, params.cifar10_std) 
    
    ###### OPTIMIZER #######
    def configure_optimizers(self, verbose=False):
        return configure_optimizers_(self, verbose)

    ######### LOSS #########
    def compute_loss(self, logits, y, x):
        # all are sums that will later be divided across the epoch. 
        # loss remains at a vector
        if self.params.classification:
            loss = F.cross_entropy(logits, y, reduction='none')
        else:
            loss = F.mse_loss(logits, x.flatten(start_dim=1), reduction='none').mean(dim=1)

        return loss

    def learn_k_reinforce_loss(self, loss, reinforce_opt):
        reinforce_opt.zero_grad()
        reinforce_loss = (-self.top_k.k_log_probs * loss).mean()
        self.manual_backward(reinforce_loss)
        reinforce_opt.step()

    def extra_loss_terms(self, neuron_activations):
        # these are scalars
        extra_loss = 0
        # regularization if enabled
        if self.params.l1_loss_weight is not None or self.params.l2_loss_weight is not None:
            extra_loss += regularization_terms(self.parameters(), self.params.l1_loss_weight, self.params.l2_loss_weight)

        if self.params.activation_l1_coefficient is not None: 
            extra_loss += self.params.activation_l1_coefficient*neuron_activations.abs().sum(dim=1).mean()

        return extra_loss

    def extra_loss_metrics(self, logits, y, x):
        # returns a dictionary with the extra loss metrics
        elm = dict()
        if "accuracy" in self.params.metrics_to_log: 
            # then compute accuracy too: 
            acc = (y == torch.argmax(logits, dim=1)).type(torch.float).sum().detach()
            elm['accuracy'] = acc.item()
        return elm


    def on_fit_start(self):
        print("Current epoch at fit start is: ", self.trainer.current_epoch)
        push_train_batch_end_commands_to_network(self)

        print("Model is:", self)

    ###### TRAIN + VAL #######
    def on_train_start(self, *args):
        # Need to update params in cases where loading in model.
        #import ipdb 
        #ipdb.set_trace() 

        if self.params.use_wandb:

            self.logger.experiment.config.update(
            self.params.__dict__, allow_val_change=True)

        # making the logger for train and val performance metrics. 
        self.performance_logger = dict()
        for prefix in ['train', 'val']:
            self.performance_logger[prefix] = {}

    def on_epoch_start(self, *args):
        # create a logger and set it to train or val. 
        pass 

    def model_step(self, data_batch, batch_idx, dataloader_idx=None):
        # shared by train and validation steps: 
        if self.training: 
            opt = self.optimizers()
            if type(opt) is list and len(opt)==2:
                opt, extra_opt = opt
            opt.zero_grad()

        x, y = data_batch
            
        logits, latent_code = self.forward(x)
        # loss here is still a vector of the batch size. 
        loss = self.compute_loss(logits, y, x)
        loss += self.extra_loss_terms(latent_code)
        logged_loss = loss 

        loss_metrics = {"loss":logged_loss.sum().item()}
        loss_metrics.update( self.extra_loss_metrics(logits, y, x) )
        store_loss_metrics(self.performance_logger, self.training,
            loss_metrics, len(x) )

        if self.training: 
            # loss up to this point is a sum (both main loss and extra terms) (easier for logging data per epoch. thus here I want to divide it by batch size. )

            self.manual_backward(loss.sum()/len(x))
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.params.gradient_clip)
            opt.step()

            if "k_approach" in vars(self.params) and self.params.k_approach == "LEARN_K_REINFORCE":
                self.learn_k_reinforce_loss( loss.detach(), extra_opt)

        else: 
            validation_logging(self, logits, y, x, batch_idx, )

        return loss

    def on_train_epoch_start(self, *args):
        pass 
            
    def training_step(self, train_batch, batch_idx):
        loss = self.model_step(train_batch, batch_idx)
        return loss 

    def on_train_batch_end(self, *args):
        push_train_batch_end_commands_to_network(self)

    def on_train_epoch_end(self, *args):
        log_loss_metrics(self)
        push_epoch_end_commands_to_network(self)

        if self.lr_scheduler: 
            self.lr_scheduler.step()

    def on_validation_epoch_start(self, *args):
        pass 

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        
        if self.params.noise_off_during_eval:
            self.noise_layer.turn_off_diffusion_noise()

        loss = self.model_step(val_batch, batch_idx, dataloader_idx=dataloader_idx)

        if self.params.noise_off_during_eval:
            self.noise_layer.turn_on_diffusion_noise()

        return loss 

    def on_validation_epoch_end(self, *args):
        # need to write out the performance loggers. 
        log_loss_metrics(self)
        
    def on_epoch_end(self, *args): 
        pass 
        # forcing all data to be logged up. 
        # called for train and validation. 

        if self.params.logger:
            wandb.log({"epoch":self.trainer.current_epoch},commit=True)

    def on_exception(self, *args):
        push_train_batch_end_commands_to_network(self)

    def on_save_checkpoint(self, ckpt):
        push_train_batch_end_commands_to_network(self)
        ckpt["hyper_parameters"] = vars(self.params)


