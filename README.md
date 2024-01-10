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




## Connection using VM
### Open a first terminal
#### 1. Connect to a machine that has GPU (for now only 025 seems to work)
```
ssh <user-name>@sl-tp-br-025.imta.fr 
```
#### 2. Activate turb envir
```
conda activate turb
```
#### 3. "Connection" to jupyter 
```
jupyter notebook --no-browser --port=8888
```
### Open a second terminal 
```
ssh -N -L localhost:1234:localhost:8888 <user-name>@sl-tp-br-025.imta.fr 
```
### Open your browser
#### 1.
```
http://localhost:1234
```
#### 2. (You might have to do this step)
Copy the token shown in the first terminal if asked 





## Dataset
### Generate the data 
```
python ./src/data/generate_data.py
```
Note: Modify path in the file if needed

### Explore

Once you have generated the data, you can start training the NN model. 
To do so, use the notebook regressor_training.ipynb in src/nn. 

Things you need to modify in the notebook :
- in the 3rd cell, we use pctes.DATAPATH to get the path where the data (MRW file) is saved. Go to src/ctes to add your own path and call it in the notebook.
- check that the device is using 'cuda' and not the cpu so that calculations don't last too long
- modify the different parameters (without forgetting to update the model name) 

Once the model in done training, you can go to the regressor_results.ipynb file to see the results on the test set. 
To do so, you need to retrieve the timestamp of the generated model by going to the data (not in src !) folder. In theory you should have 3 files for your training : losses, model, hyperparameters with all the same timestamp. Copy paste this timestamp in the notebook (2nd cell). As for the training notebook, you also need to update path via pctes (in 2nd cell). 

(What good results should look like : training and validation losses converge towards the same value and the Actual vs Preds graphs are "diagonal") 



Another training notebook was added : regressor_training_general.ipynb. The goal is to more easily launch several training one after the other by calling the function in successive cells and letting the VM open for several hours (for example during the night).





