import os
import sys
import copy
'''if 'exp_commands' not in os.getcwd():
    os.chdir('exp_commands/')
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append('../py_scripts')
if 'exp_commands' in os.getcwd():
    os.chdir('..')'''
########### EXPERIMENTS TO RUN ##############

"""
Ablations of diffusion models.
"""

main_title = "DiffLRs"

settings_for_all = dict(
    epochs_to_train_for = 1000,
    model_name= "DIFFUSION_FFN",
    dataset_name = "Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10", #CIFAR10,#"Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10", #!!

    opt='Adam',#SparseAdam',
    lr=0.0001,#0.0001,#0.15,
    
    batch_size=1024,#1000, 
    dataset_size = None,#1000,

    nneurons=[10000],

    nconvergence_steps=1,
    diffusion_noise=0.3,
    adjust_diffusion=False,

    use_projection_matrix=False,
    project_before_noise = False,
    
    use_bias_hidden = True,
    use_bias_output = True, 
    use_shared_bias = False, 

    transpose=False, 
    classification=False, 

    #k_min=50, 
    #k_transition_epochs = 100,
    #k_approach = "LINEAR_DECAY_MASK",

    #all_positive_weights = False, 
    #norm_addresses=True, 
    #norm_values = True, 
    #norm_activations=False,

    preprocessing_type = "SIMPLE", 
    log_receptive_fields = False,
    
    log_model_predictions=False, 
    
)

modifications = dict(

    transpose=False, 

)

settings_for_all.update(modifications)

# sigma={settings_for_all['diffusion_noise']}
name_suffix = f"{main_title}_{settings_for_all['opt']}_lr{settings_for_all['lr']}_datas={settings_for_all['dataset_size']}_{settings_for_all['nconvergence_steps']}steps_{settings_for_all['nneurons'][0]}Neurons_projM={settings_for_all['use_projection_matrix']}_nlayers{len(settings_for_all['nneurons'])}"  
# _kmin={settings_for_all['k_min']}

exp_list = [
    dict(
        
        test_name= "0.6_ffn",
        diffusion_noise=0.6,
        lr=0.01
    ),

    dict(
        
        test_name= "0.3_ffn",
        diffusion_noise=0.3,
        lr=0.01
    ),
    
    dict(

        test_name= "0.6_ffn",
        diffusion_noise=0.6,
        lr=0.001
    ),

    dict(

        test_name= "0.3_ffn",
        diffusion_noise=0.3,
        lr=0.001
    ),

    dict(

        test_name= "0.6_ffn",
        diffusion_noise=0.6,
        lr=0.00001
    ),

    dict(

        test_name= "0.3_ffn",
        diffusion_noise=0.3,
        lr=0.00001
    ),
    

]

"""
    dict(
        
        test_name= "100N_ffn",
        nneurons=[100],
    ),
    dict(
        
        test_name= "1_000N_ffn",
        nneurons=[1000],
    ),
    dict(
        
        test_name= "10_000N_ffn",
        nneurons=[10000],
    ),
    dict(
        
        test_name= "100_000N_ffn",
        nneurons=[100000],
    ),
    =============================

    ---------------

    dict(
        
        diffusion_noise=0.0,
        test_name= "0sigma_ffn",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        diffusion_noise=0.0,
        test_name= "0sigma_sdm",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        diffusion_noise=0.0,
        test_name= "0sigma_sdm_nol2",
        norm_addresses=False, 
        norm_values = False, 
        norm_activations=False,
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        diffusion_noise=0.0,
        test_name= "0sigma_sdm_flat",
        k_approach = "FLAT_MASK",
        
    ),

    dict(
        
        
        test_name= "ffn",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        
        test_name= "sdm",
    ),

    dict(
        
        diffusion_noise=0.5,
        test_name= "0.5sigma_ffn",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        diffusion_noise=0.5,
        test_name= "0.5sigma_sdm",
    ),


#####

    
-----



dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=100,
        test_name= "sdm_flat_k=100",
        k_approach = "FLAT_MASK",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=50,
        test_name= "sdm_flat_k=500",
        k_approach = "FLAT_MASK",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=20,
        test_name= "sdm_flat_k=50",
        k_approach = "FLAT_MASK",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=1000,
        test_name= "sdm_flat_k=2000",
        k_approach = "FLAT_MASK",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=3000,
        test_name= "sdm_flat_k=5000",
        k_approach = "FLAT_MASK",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=8000,
        test_name= "sdm_flat_k=8000",
        k_approach = "FLAT_MASK",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        test_name= "sdm_norm_all",
        norm_addresses=True, 
        norm_values = True, 
        norm_activations=True,
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        test_name= "sdm_norm_addr_and_vals",
        norm_addresses=True, 
        norm_values = True, 
        norm_activations=False,
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        test_name= "sdm_norm_address",
        norm_addresses=True, 
        norm_values = False, 
        norm_activations=False,
    ),


    ----

    dict(
        
        test_name= "0.1_ffn",
        diffusion_noise=0.1,
    ),
    dict(
        
        test_name= "0.3_ffn",
        diffusion_noise=0.3,
    ),
    dict(
        
        test_name= "0.6_ffn",
        diffusion_noise=0.6,
    ),
    
    dict(
        
        test_name= "1.0_ffn",
        diffusion_noise=1.0,
    ),
    dict(
        
        test_name= "1.5_ffn",
        diffusion_noise=1.5,
    ),

"""

if __name__ == '__main__':
    print(len(exp_list))
    # copy out this script to elsewhere
    os.system(f"cp experiment_generator.py exp_commands/{main_title}_{settings_for_all['model_name']}.py")
    sys.exit(0)