# Running multiple experiments across GPUs
import ray
import torch
import matplotlib.pyplot as plt
from py_scripts import * 
import pytorch_lightning as pl
import wandb
import multiprocessing
from torch import nn
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import time 
import os 
import subprocess as sp
import argparse
import copy 
from py_scripts import *
#from exp_commands.temp_slurm_jobs import *
#from exp_commands import *

import importlib

args = None  # makes args into a global variable that is set during the experiment run


#I NEED TO SYMLINK THIS. 

ncpus_per_worker = 1  # number of cpus each job is allocated
# assuming they are all quite homogenous: 

####################
memory_per_job = 4000 #8000  # in MB. How much memory to assign to each job.
#####################

gpu_capacity = 16160 # 
enforce_memory_per_job = False  # false enables dynamic memory allocation but I will still estimate memory per job to not overload a given GPU.
mem_threshold_to_use_gpu = gpu_capacity-memory_per_job  # in MB. Must be below to use the GPU.
activity_threshold = 50 #out of 100
ncpus_to_allocate = 16
# Getting the GPUs that are free
gpus_to_use, gpu_str = get_free_gpus(mem_threshold_to_use_gpu, activity_threshold, use_all=True)
ngpus = len(gpus_to_use)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

fraction_of_gpu = memory_per_job / gpu_capacity
print("Fraction of each GPU to use:", round(fraction_of_gpu,2))

print("3 Seconds to Cancel")
time.sleep(3)

if enforce_memory_per_job:
    for new_gpu_ind in range(ngpus):
        torch.cuda.set_per_process_memory_fraction(fraction_of_gpu, new_gpu_ind)

print('number of gpus being used', ngpus)

# Use local_mode = True to debug. 
ray.init(local_mode=False, num_cpus=ncpus_to_allocate, num_gpus=ngpus)
@ray.remote(num_gpus=fraction_of_gpu, num_cpus=ncpus_per_worker, max_calls=1)
def start_experiment(exp_ind):

    # RUNNING THE EXPERIMENT THAT HAS BEEN LOADED IN
    # SETTING GLOBAL PARAMETERS FOR EVERY EXPERIMENT HERE IN CASE THEY WERE NOT SET FOR THE SPECIFIC ENTRY.

    exp_settings = init_exp_settings(exp_ind,job_script)

    model, data_module, params, callbacks, checkpoint_callback = compile_experiment(exp_settings, ncpus_per_worker)

    temp_trainer = pl.Trainer(
                        logger=params.logger,
                        max_epochs=params.epochs_to_train_for,
                        num_sanity_val_steps = 0,
                        check_val_every_n_epoch=params.check_val_every_n_epoch,
                        gpus=1, 
                        auto_select_gpus=True,
                        enable_progress_bar = False,
                        callbacks=callbacks,
                        checkpoint_callback = checkpoint_callback, 
                        detect_anomaly=False,
                        )
    temp_trainer.fit(model, data_module, ckpt_path=params.fit_load_state)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_script", type=str, default=None, required=False, help="The job script.")

    args = parser.parse_args()

    if not args.job_script:
        args.job_script = "experiment_generator"

    # "exp_commands."+
    job_script = importlib.import_module(args.job_script)
        #from exp_commands.args.job_script import *

    os.system(f"cp {args.job_script}.py exp_commands/{job_script.main_title}_{job_script.settings_for_all['model_name']}.py")

    out = ray.get([start_experiment.remote(exp_ind) for exp_ind in range(len(job_script.exp_list)) ])
    ray.shutdown()