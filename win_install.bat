@echo off
set env_name=%1
if "%1" == "" (
    set env_name="turb"
)
echo [TURBNNGEN] Creating conda environment %env_name% ...
CALL conda create --name %env_name% python=3.11 --yes
echo [TURBNNGEN] Activating conda environnement %env_name% ...
CALL conda activate %env_name%
echo [TURBNNGEN] Installing pytorch ...
CALL conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
echo [TURBNNGEN] Installing conda requirements ...
CALL conda install conda_requirements --yes
echo [TURBNNGEN] Installing pip requirements ...
CALL pip install -r pip_requirements.txt
echo [TURBNNGEN] Installing source code ...
call python -m pip install -e .