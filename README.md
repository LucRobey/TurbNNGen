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
