# PETALS

This is the official repository for PETALS (Plant-state ExTrAction using Latent Spaces). The accompanying paper can be 
found here:

## Overview
The different methods implemented are in the method packages: 
- `method_t` - Absolute method using the wall-clock age (t) as targets.
- `method_contrast` - Contrast method using contrastive learning method.
- `method_order` - Order method using the temporal ordering for training.
- `method_classic` - HSV colour thresholding
- `method_dimred` - Classical dimensionality reduction methods: PCA, t-SNE, UMAP
- `method_vae` - Neural dimensionality reduction method: VAE 

Each method has similar structures:
- `models.py` - Stores the models used 
- `predictor.py` - Wraps around the models and provides a consistent API
- `run.py/run_expt.py` - Runs experiments
- `utils.py` - Utilities such as dataloaders, routing to the correct models, etc.

Many different models can be defined inside `models.py`. Each has to be given a name and registered in 
the `init_model` function inside the method's `utils.py` file. Similarly many dataloaders can be defined and registered 
in the `init_data` function inside `utils.py`.

## Usage
Signup to weights and biases. The first time you run code here you will be 
prompted for the wandb API key.

### Train Models
Create a project on weights and biases, this will be used to store the trained models/visualise training, etc. <br>
Example `project_name` name: `Plant State`

Inside the `run.py/run_expt.py` file specify the `config` and corresponding `project_name` (the one you made 
previously). Run these files in the root of the directory.


### Evaluate Models
Create a new project, this will be for keeping track of model evaluations. <br>
Example `project_name` name: `Evaluation`

Inside the `run_expt.py` file inside the `evaluation` package, specify the `config` that will be used for evaluation.

`source_project_name` corresponds to the project name used for training models.

`source_expt_name` corresponds to the experiment name from the source project that you want to evaluate.

`method` corresponds to the method used for the specified `source_expt_name` <br>
(contrast, t, order, colour_threshold, dimred)
