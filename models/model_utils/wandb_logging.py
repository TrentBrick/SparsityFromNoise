import wandb 
import torch 

def time_to_log_images(self, batch_idx):
    return (self.trainer.current_epoch+1)%self.params.log_image_data_every_n_epochs==0 and batch_idx==0

def validation_logging(self, logits, y, x, batch_idx):
    # logging examples of classification or reconstruction here. Need to do it here where I have access to the logits
    if time_to_log_images(self, batch_idx):

        if self.params.log_model_predictions:
            if self.params.classification:
                save_classification_prediction(self, x, y, logits)
            else:  
                save_image_reconstruction(self, x, y, logits)

        if self.params.log_receptive_fields:
            save_receptive_fields(self)

def increment_dict(dic, key, val):
    if key in dic.keys():
        dic[key]+=val 
    else:
        dic[key]=val 
    #return dic

def store_loss_metrics(performance_logger, is_training, metrics, batch_size):
    # reverses the mean operation so apply it later at the end.
    prefix = "train" if is_training else "val"
    for k, v in metrics.items():
        increment_dict(performance_logger[prefix], k, v)
    increment_dict(performance_logger[prefix], 'batch_size', batch_size)
            
def log_loss_metrics(self):
    # adjust by batch size and alter name of each item before logging out to wandb. Resets original too
    # runs for train and val separately!
    prefix = "train" if self.training else "val"
    to_log = dict()
    for k, v in self.performance_logger[prefix].items():
        if "batch_size" in k:
            continue

        if "split" in k: 
            # get the batch size tracker for just this task. 
            to_log[f'{prefix}/{k}'] = v/self.performance_logger[prefix][f"batch_size_split_{k.split('_')[-1]}"]
        else: 
            to_log[f'{prefix}/{k}'] = v/self.performance_logger[prefix]['batch_size']

    # reset performance logger.
    self.performance_logger[prefix] = {'batch_size':0} 

    # log everything out
    log_wandb(self, to_log)

def log_wandb(self, dic, commit=False):
    if self.params.logger and self.params.use_wandb:
        assert type(dic) == dict, "Need to input a dictionary"
        # Custom function that adds the epoch (and whatever else in the future.) 
        # Also allows for commit to be false to only force it later. 
        dic['epoch'] = self.trainer.current_epoch
        dic['global_step'] = self.trainer.global_step
        wandb.log(dic,commit=commit)
    else: 
        if "train/loss" in dic.keys() or "val/loss" in dic.keys():
            print(dic)

def log_image(self, weights_to_plot, title, sub_title, inds=None, cnn_weights=False, num_images=None):
    if cnn_weights:
        plot_dim = weights_to_plot.shape[-1]
        if num_images is None: 
            num_images = self.params.num_cnn_receptive_field_imgs
    else:
        plot_dim = self.params.img_dim
        if num_images is None: 
            num_images = self.params.num_receptive_field_imgs
    if inds is None:
        inds = range(num_images)
    log_wandb(self,
        {
            title: [
                wandb.Image(
                    (
                        weights_to_plot[i].reshape(
                            plot_dim, plot_dim
                        )
                        if self.params.nchannels == 1
                        else weights_to_plot[i].reshape(
                            self.params.nchannels,
                            plot_dim,
                            plot_dim,
                        )
                    ),
                    caption=f"{sub_title} #{i}",
                )
                for i in inds
            ]
        }
    )

def save_classification_prediction(self, x, y, logits):
    log_wandb(self,
        {
            "Classification Examples": [
                wandb.Image(
                    x_i,
                    caption=f"Ground Truth: {y_i}\nPrediction: {y_pred}",
                )
                for x_i, y_i, y_pred in list(
                    zip(
                        x[: self.params.num_task_attempt_imgs],
                        y[: self.params.num_task_attempt_imgs],
                        torch.argmax(
                            logits[: self.params.num_task_attempt_imgs], dim=1
                        )
                    )
                )
            ]
        }
    )

def save_image_reconstruction(self, x, y, logits):
    # ground truth:
    log_wandb(self,
        {
            "Ground Truth Examples": [
                wandb.Image(
                    x_i,
                    caption=f"Label: {y_label}",
                )
                for y_label, x_i in zip(
                    y[: self.params.num_task_attempt_imgs],
                    x[: self.params.num_task_attempt_imgs]
                )
            ]
        }
    )
    # generation:
    log_wandb(self,
        {
            "Reconstruction Examples": [
                wandb.Image(
                    y_pred,
                    caption=f"Label: {y_label}",
                )
                for y_label, y_pred in zip(
                    y[: self.params.num_task_attempt_imgs],
                    logits[: self.params.num_task_attempt_imgs]
                )
            ]
        }
    )

def save_receptive_fields(self):

    if self.params.cnn_model:      
        self.log_image( self.features[0].weight.data, "Random CNN Receptive Fields (Addresses)", "Neuron", cnn_weights=True)
        # log filters here!! 
    else: 
        model_weights = self.X_a.weight.data
        # log the sdm layer (if it exists)
        # if CNN is on then SDM is not only the first layer and so it doesnt make much sense to visualize it. 
        self.log_image( model_weights, "Random Receptive Fields (Addresses)", "Neuron")

        vals, inds = torch.topk( self.X_a_activations , 5, dim=0, sorted=False)

        self.log_image( model_weights, "Most Active Receptive Fields (Addresses)", "Most Active Neuron", inds=inds)