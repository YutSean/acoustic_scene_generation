This code is modified from the implementation of MelNet: https://github.com/Deepest-Project/MelNet

# How to train
Create config .yaml file in config directory and run ``` python train.py -c [config YAML file path] -n [name of run] -t [tier number] -b [batch size]```

# How to sample
Create inference .yaml file in config directory and run ```python inference.py -c [config YAML file path] -p [inference YAML file path] -t [timestep of generated mel spectrogram] -n [name of sample]```
