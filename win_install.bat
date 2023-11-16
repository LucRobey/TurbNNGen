conda create --name turb python=3.11 --yes
conda activate turb
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
conda install conda_requirements --yes
pip install pip_requirements.txt --yes