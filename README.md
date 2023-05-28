# Noise Transforms Feed-Forward Networks into Sparse Coding Networks

This is the codebase behind the paper `Noise Transforms Feed-Forward Networks into Sparse Coding Networks`.

We provide code sufficient to reproduce all of our experiments.

Follow these steps: 

## 1. Set up a new Conda environment using: 

```
conda create --name NoiseSparsity python=3.9
conda activate NoiseSparsity
conda install pip
pip install --upgrade pip
pip install --upgrade setuptools
```

## 2. Clone this github repo and install its requirements: 

```
git clone https://github.com/anon8371/AnonPaper2.git
cd AnonPaper2
pip install -r requirements.txt
```

You may be able to use other versions of the libraries found in `requirements.txt` but no promises. 

## 3. Setup Wandb

Make an account [here](https://wandb.ai/home) and put the details into `test_runner.py` under `wandb_logger=`. 

Otherwise, set `use_wanbd=False` to run the models but limited outputs will be printed. 

## 4. Run `python test_runner.py` that will by default run an ReLU model on CIFAR10 latents with 0.8 noise. 

See `exp_commands/` for all parameters used to run all of our experiments. 

You can put these parameters into `test_runner.py` to run them and fill in `load_path=` with a trained model. To reproduce all of our results we recommend using a job parallelizer like Ray or SLURM to run each experiment as a different job. See `ray_runner.py` for our use of ray.

---

# Code Base Overview

* `models/` - has all models.
    * `models/Base_Model.py` - has the PytorchLightning backbone that all models inherit. 
    * `models/Diffusion_Models.py` - has the Diffusion model backbone and two different models: Feedforward, SDM.  SDM models that implement some form of Top-K and have built in L2 normalization of weights, ability to enforce all positive weights, etc. All models use the `TrackedMLPLayer.py` that tracks many different metrics. 
    * `models/InhibCircuit.py` - Implements explaining away and facilitation. 
    * `models/TopK_Act.py` - Top-K activation function that implements all of the variants considered. 
    * `models/TrackedActFunc.py` - Tracks metrics such as the number of dead neurons and number of neurons active for an input. 
* `notebooks/` - ipynb notebooks to analyze the different models. 
* `py_scripts/` - handle model and data parameters, additional helper scripts for training. 
* `data/` - We provide the CIFAR10 ConvMixer latent embeddings. 
* `data/data_preprocessing/` - scripts to process and make the default CIFAR10 and other datasets load faster. 
* `ray_runner.py` - runs a number of Ray jobs in parallel. Pulls experiment details from `experiment_generator.py`

