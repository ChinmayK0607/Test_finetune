#!/bin/bash

# Conda installation
conda install pytorch=2.2.0 pytorch-cuda=12.1 cudatoolkit xformers -c pytorch -c nvidia -c xformers -y

# Pip installations
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "torch<2.3.0" "xformers<0.0.26" trl peft accelerate bitsandbytes
pip install streamlit pandas
pip install tensorboard
