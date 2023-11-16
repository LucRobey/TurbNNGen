# TurbNNGen
projet ProCom A3 imt

## Pre-requirements
* conda >= 23.5.2

## Installation
### Raw installation commands
```
conda create --name turb python=3.11 --yes
```
```
conda activate turb
```
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
```
```
conda install conda_requirements --yes
```
```
pip install -r pip_requirements.txt
```
```
python -m pip install -e .
```
### Windows installation script
```
win_install.bat
```

### Add new dependencies
Add new dependencies installed with `conda` to `conda-requirements.txt`  
Add new dependencies installed with `pip` to `pip-requirements.txt`

### Activate environnement
```
conda activate turbnn
```

### Deactivate environnement
```
conda deactivate
```

## Dataset
### Generate
```
python ./data/generate_data.py
```
Note: Modify path in the file if needed

### Explore
```
jupyter notebook ./data/MRW.ipynb
```
