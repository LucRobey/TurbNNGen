# TurbNNGen
projet ProCom A3 imt

## Pre-requirements
* conda >= 23.5.2

## Installation
```
conda create --name turbmm python=3.11
```

```
conda activate turbnn
```

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Add new dependencies
After adding new depedencies, make sure to save new environnement packages executing the following line:
```
conda list --explicit > ENV.txt
```

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

### Explore
```
jupyter notebook ./data/MRW.ipynb
```
