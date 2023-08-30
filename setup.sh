

# Create environment

mamba create -y -n restorator python=3.9

mamba activate restorator

# Install dependencies

mamba install -y -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*
