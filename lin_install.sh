#!/bin/bash

# chmod u+x <this filename>

if [ "$#" -eq 0 ] 
then
	env_mode="--name"
	env_value="turb"
else
	env_mode=$1
	env_value=$2
fi

echo "[TURBNNGEN] ($CONDA_DEFAULT_ENV) Creating conda environment $env_mode $env_value ..."
conda create $env_mode $env_value python=3.11 --yes
echo "[TURBNNGEN] ($CONDA_DEFAULT_ENV) Activating conda environnement $env_value ..."
source ~/miniconda3/bin/activate $env_value
echo "[TURBNNGEN] ($CONDA_DEFAULT_ENV) Installing pytorch ..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
echo "[TURBNNGEN] ($CONDA_DEFAULT_ENV) Installing conda requirements ..."
conda install --file conda_requirements.txt --yes
echo "[TURBNNGEN] ($CONDA_DEFAULT_ENV) Installing pip requirements ..."
pip install -r pip_requirements.txt
echo "[TURBNNGEN] ($CONDA_DEFAULT_ENV) Installing source code ..."
python -m pip install -e .

