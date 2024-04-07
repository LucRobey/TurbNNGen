# TurbNNGen

## Download

### HTTPS
```
git clone https://github.com/LucRobey/TurbNNGen.git
```
### SSH
```
git clone git@github.com:LucRobey/TurbNNGen.git
```
### Github CLI
```
gh repo clone LucRobey/TurbNNGen
```
## Prerequisites
* Conda (v23.5.2)

## Pre-installation
Clean up environment
```
conda clean --all
```
Choose directory for library downloads
```
conda config --add pkgs_dirs <path-downloading-dir>
```

## Installation
Three installation methods are available:
### Linux Script
Grant execution permission to the script
```
chmod u+x lin_install.sh
```
Run the script (2 options):
**Specify an environment name**
```
./lin_install.sh --name <env-name>
```
**Specify installation environment location**
```
./lin_install.sh --prefix <env-installation-path>
```
### Windows Script
Run the script (2 options):
Specify an environment name
```
./win_install.bat--name <env-name>
```
Specify installation environment location
```
./win_install.bat --prefix <env-installation-path>
```

### Step-by-step
Create Conda environment (2 options)
**Specify an environment name**
```
conda create --name <env-name> python=3.11 --yes
```
**Specify installation environment location**
```
conda create --prefix <env-installation-path> python=3.11 --yes
```
Activate Conda environment (2 options)
**Specify environment name**
```
conda activate <env-name>
```
**Specify installation environment location**
```
conda activate <env-installation-path>
```
Install PyTorch and other associated libraries
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
```
Install additional libraries with Conda
```
conda install --file conda_requirements.txt --yes
```
Install additional libraries with pip
```
pip install -r pip_requirements.txt
```
Install source code as a library
```
python -m pip install -e .
```
## Activation of Environment
Activate Conda environment (2 options)
**Specify environment name**
```
conda activate <env-name>
```
**Specify installation environment location**
```
conda activate <env-installation-path>
```
## Data Generation
Kolmogorov statistics of generated data are automatically normalized to be between 0 and 1, facilitating subsequent training tasks.

### Initial Data Generation
```
python ./src/data/generate_data.py --datapath <path-to-save-data> --scalerpath <path-to-save-scaler>
```
Arguments:
* `datapath`: Path to the file (existing or not) to save the generated data. The save directory must exist.
* `scalerpath`: Path to the file (existing or not) to save the object that scaled the generated data. The save directory must exist.
### Generating More Data
```
python ./src/data/generate_more_data.py --datapath <path-to-data> --scalerpath <path-to-scaler>
```
Arguments:
* `datapath`: Path to the existing file containing the originally generated data.
* `scalerpath`: Path to the existing file containing the object that scaled the originally generated data.

### Data Denormalization
```
python ./src/data/denormalizer.py <path-to-data> <path-to-scaler>
```
### Data Exploration
```
jupyter notebook src/data/MRW.ipynb
```
Note: Specify the relative route to the data in the J

## Neural Networks

All information regarding the definition, training, and evaluation of neural networks can be found in the following directories:
```
src/
|-- nn/
|-- archs/
|-- losses/
|-- results/
|-- training/
```
* `src/nn/archs`: Contains various architectures.
* `src/nn/losses`: Contains definitions of loss functions.
* `src/nn/results`: Contains Jupyter notebooks to analyze the results of various trained architectures.
* `src/nn/training`: Contains Jupyter notebooks to train various architectures.

### Training

There are 5 training files:

* `src/nn/training/cnn_all_training.ipynb`: Architectures aiming to estimate all statistics (c1, c2, L, epsilon).
* `src/nn/training/cnn_c1_training.ipynb`: Architectures aiming to estimate c1.
* `src/nn/training/cnn_c2_training.ipynb`: Architectures aiming to estimate c2.
* `src/nn/training/cnn_L_training.ipynb`: Architectures aiming to estimate L.
* `src/nn/training/cnn_epsilon_training.ipynb`: Architectures aiming to estimate epsilon.

All these files can be opened in a Jupyter notebook, hyper-parameterized (following the steps specified in the notebooks), and executed.

For example:
```
jupyter notebook src/nn/training/cnn_all_training.ipynb
```
Note: It is necessary to specify the relative path to the data in the Jupyter notebook. This action is also possible from the following file:
* `src/nn/path_ctes.py`
```
DATAPATH   = "../../../data/MRW.npz"
SCALERPATH = "../../../data/scaler.joblib"
```
After completing the training, the following files can be found in the following directory:
```
data/
|-- models/
   |-- model_<model>_<timestamp>_<best/last>.pt
   |-- hyperparameters_<model>_<timestamp>.npz
   |-- losses_<model>_<timestamp>.npz
```
* `<model>`: Name of the class defining the architecture.
* `<timestamp>`: Execution timestamp.
* `model_<model>_<timestamp>_<best/last>.pt`: Model weights.
  * `best`: Weights that resulted in the lowest loss concerning the validation dataset.
  * `last`: Weights that resulted in the last loss.
* `hyperparameters_<model>_<timestamp>.npz`: Hyperparameters used to train the model.
* `losses_<model>_<timestamp>.npz`: Evolution of training and validation loss.

### Tests
There are 5 result files:

* `src/nn/results/cnn_all_results.ipynb`: Architectures aiming to estimate all statistics (c1, c2, L, epsilon).
* `src/nn/training/cnn_c1_training.ipynb`: Architectures aiming to estimate c1.
* `src/nn/training/cnn_c2_training.ipynb`: Architectures aiming to estimate c2.
* `src/nn/training/cnn_L_training.ipynb`: Architectures aiming to estimate L.
* `src/nn/training/cnn_epsilon_training.ipynb`: Architectures aiming to estimate epsilon.

All these files can be opened in a Jupyter notebook.

For example:
```
jupyter notebook src/nn/results/cnn_all_results.ipynb
```
Once opened, the builder, timestamp, and data_path fields must be filled to specify the training results to visualize.

For example:

```
builder   = CNN_ALL_VDCNNFRW_M18
timestamp = "2024_02_08__09_32_10"
data_path = pctes.DATAPATH
```
`builder`: Class defining the architecture whose results will be analyzed.
`timestamp`: Execution timestamp of the training. This can be identified at the time of training execution or in the filenames in data/models/.
`data_path`: Relative data path. This path can also be specified from the following file:
* `src/nn/path_ctes.py`
```
DATAPATH   = "../../../data/MRW.npz"
SCALERPATH = "../../../data/scaler.joblib"
```

The most important results to visualize in this notebook are the following ones.

#### Loss Plot
#### Prediction Summary
```
Total Test MSE = 0.0138
Test MSE for each output:
c1: 0.0019
c2: 0.0167
L: 0.0286
EPSILON: 0.0080
```
#### Prediction Distribution Plot
#### Correlation Matrix of Predictions
