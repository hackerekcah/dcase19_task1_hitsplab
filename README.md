# dcase19_task1_hitsplab
Dcase2019 Task1 (Acoustic Scene Classification) Challenge Code.

# Introduction


# Install
* create new env, and install packages in requirements.txt
```
conda create --name <env> --file requirements.txt
```
*if want to run jupyter notebook examples, install a kernelspec for env
```
conda install jupyter ipykernel
python -m ipykernel install --user --name <env> --display-name 'python3.6(<env>)'
```
# data_manager
* create file data_manager.cfg under data_manager/
* specify `dev_path` to point to dcase2019 Task1 SubTaskB development dataset
* specify `lb_path` to point to dcase2019 Task1 SubTaskB leadboard dataset
* specify `eva_path` to point to dcase2019 Task1 SubTaskB evaluation dataset
```
[DEFAULT]

[dcase19_taskb]
dev_path = /PATH TO .../dcase2019_task1_baseline/datasets/TAU-urban-acoustic-scenes-2019-mobile-development/
lb_path = /PATH TO .../dcase2019_task1_baseline/datasets/TAU-urban-acoustic-scenes-2019-mobile-leaderboard/
eva_path = /PATH TO .../dcase2019_task1_baseline/datasets/TAU-urban-acoustic-scenes-2019-mobile-evaluation/

[logmel]
sr = 44100
n_fft = 4096
hop_length = 1024
n_mels = 128
fmax = 22050
```

* extract feature and store it in .h5 file
```
# extract feature for development set
# will generate .h5 files under data_manager/data19_h5
python data_manager/dcase19_taskb.py

# extract mean and variance for development set
# will generate .h5 files under data_manager/data19_h5
python data_manager/dcase19_standrizer.py

# extract feature for leaderboard data
# will generate .h5 files under data_manager/data19_h5
python data_manager/dcase19_lb_manager.py

# extract feature for evaluation data
# will generate .h5 files under data_manager/data19_h5
python data_manager/dcase19_eva_manager.py
```

# Experiments
* open `jupyter notebook` or `jupyter lab`
* try and run experiments notebooks under jupyter_nbs/
* all experiments configurations are listed in detail, any one can try to reproduce it


