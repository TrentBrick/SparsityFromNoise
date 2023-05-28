import os
import sys
import copy
import torch 

settings_for_all = dict(
    epochs_to_train_for = 6000,
    model_name= "DIFFUSION_FFN",
    dataset_name = "Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10", #CIFAR10,#"Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10", #!!

    opt='Adam',#SparseAdam',
    lr=0.0001,#0.0001,#0.15,
    
    batch_size=1024,#1000, 
    dataset_size = None,#1000,

    nneurons=[10000],

    diffusion_noise=0.0,
    adjust_diffusion=False,

    use_projection_matrix=False,
    project_before_noise = False,
   
    use_shared_bias = False, 

    transpose=False, 
    classification=False, 

    log_receptive_fields = False,
    log_model_predictions=False, 
    
)

main_title = "Another6KEpochs_CIFAR10"

epoch_and_step = "epoch=5989-step=293510.ckpt"#"epoch=799-step=39200.ckpt"
exp_name_path = "ReconCIFAR10Long_EvenMore_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1"#"Poisson_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1" #"Baseline_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1"

load_all_models = True

modifications = dict(

    #act_func = torch.nn.Sigmoid(),
    #dataset_name = "CIFAR10",

)

settings_for_all.update(modifications)

# sigma={settings_for_all['diffusion_noise']}
name_suffix = f"{main_title}_{settings_for_all['opt']}_lr{settings_for_all['lr']}_datas={settings_for_all['dataset_size']}_{settings_for_all['nneurons'][0]}Neurons_projM={settings_for_all['use_projection_matrix']}_nlayers{len(settings_for_all['nneurons'])}"  
# _kmin={settings_for_all['k_min']}

init_exp_list = [

    dict(
        test_name= "0.0",
        diffusion_noise=0.0,
    ),
    dict(
        test_name= "0.05",
        diffusion_noise=0.05,
    ),
    dict(
        test_name= "0.1",
        diffusion_noise=0.1,
    ),
    dict(
        test_name= "0.3",
        diffusion_noise=0.3,
    ),
    dict(
        test_name= "0.8",
        diffusion_noise=0.8,
    ),
    dict(
        test_name= "1.5",
        diffusion_noise=1.5,
    ),
    dict(
        test_name= "3.0",
        diffusion_noise=3.0,
    ),
    dict(
        test_name= "10.0",
        diffusion_noise=10.0,
    ),


    
    
]

if load_all_models: 


    exp_list = []

    #for rand_seed in [10, 55, 78]:
    for e in init_exp_list: 
        temp = copy.deepcopy(e)
        temp['load_path'] = f"../scratch_link/Foundational-SDM/wandb_Logger/{temp['test_name']}{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
        exp_list.append( temp )

else: 
    exp_list = init_exp_list
    

print("Total number of experiments", len(exp_list))   


"""

dict(
        test_name= "0.001",
        activation_l1_coefficient=0.001,
    ),
    dict(
        test_name= "0.0001",
        activation_l1_coefficient=0.0001,
    ),

    dict(
        test_name= "0.00001",
        activation_l1_coefficient=0.00001,
    ),
    dict(
        test_name= "0.000001",
        activation_l1_coefficient=0.000001,
    ),
    dict(
        test_name= "0.0000001",
        activation_l1_coefficient=0.0000001,
    ),
    dict(
        test_name= "0.0",
        activation_l1_coefficient=0.0,
    ),

    

dict(
        test_name= "0.0",
        diffusion_noise=0.0,
    ),
    dict(
        test_name= "0.05",
        diffusion_noise=0.05,
    ),
    dict(
        test_name= "0.1",
        diffusion_noise=0.1,
    ),
    dict(
        test_name= "0.3",
        diffusion_noise=0.3,
    ),
    dict(
        test_name= "0.8",
        diffusion_noise=0.8,
    ),
    dict(
        test_name= "1.5",
        diffusion_noise=1.5,
    ),
    dict(
        test_name= "3.0",
        diffusion_noise=3.0,
    ),
    dict(
        test_name= "10.0",
        diffusion_noise=10.0,
    ),



    dict(
        test_name= "0.1",
        diffusion_noise=0.1,
    ),
    dict(
        test_name= "1.0",
        diffusion_noise=1.0,
    ),
    dict(
        test_name= "3.0",
        diffusion_noise=3.0,
    ),
    dict(
        test_name= "5.0",
        diffusion_noise=5.0,
    ),
    dict(
        test_name= "8.0",
        diffusion_noise=8.0,
    ),
    dict(
        test_name= "10.0",
        diffusion_noise=10.0,
    ),

"""

if __name__ == '__main__':
    print(len(exp_list))
    # copy out this script to elsewhere
    os.system(f"cp experiment_generator.py exp_commands/{main_title}_{settings_for_all['model_name']}.py")
    sys.exit(0)