# TurbNNGen
projet ProCom A3 imt

## Pre-requirements
* conda >= 23.5.2

## Installation

### Raw installation commands
#### 0.
Clean space for the project:
```
conda clean --all
```

Change folder for downloading packages:
```
conda config --add pkgs_dirs <path-downloading-dir>
```

#### 1.
```
conda create --name <env-name> python=3.11 --yes
```
or 
```
conda create --prefix <env-installation-path> python=3.11 --yes
```
#### 2.
```
conda activate <env-name>
```
or 
```
conda activate <env-installation-path>
```
#### 3.
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
```
#### 4.
```
conda install --file conda_requirements.txt --yes
```
#### 5.
```
pip install -r pip_requirements.txt
```
#### 6.
```
python -m pip install -e .
```
### Windows installation script
```
win_install.bat
```

### Linux installation script
```
chmod u+x lin_install.sh
```
```
lin_install.sh --name <env-name>
```
or
```
lin_install.sh --prefix <env-installation-path>
```

### Add new dependencies
Add new dependencies installed with `conda` to `conda-requirements.txt`  
Add new dependencies installed with `pip` to `pip-requirements.txt`

### Activate environnement
```
conda activate turb
```

### Deactivate environnement
```
conda deactivate
```

## Remote Connection
### Terminal 1
#### 1.
```
ssh <user-name>@<computer-name>.imta.fr
```
#### 2.
```
source ~/miniconda3/bin/activate <env-name>
```
or
```
source ~/miniconda3/bin/activate <env-installation-path>
```
#### 3.
```
jupyter notebook --no-browser --port=8888
```
### Terminal 2
```
ssh -N -L localhost:1234:localhost:8888 <user-name>@<computer-name>
```
### Browser
#### 1.
```
http://localhost:1234
```
#### 2.
Copy the token shown in `Terminal 1`

## Dataset
### Generate
```
python ./src/data/generate_data.py
```
Note: Modify path in the file if needed

### Explore
```
jupyter notebook ./src/data/MRW.ipynb
```
