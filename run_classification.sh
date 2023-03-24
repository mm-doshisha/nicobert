#!/bin/bash

source /etc/profile.d/modules.sh

module load singularitypro gcc/9.3.0 python/3.8/3.8.7 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 openmpi/4.0.5

singularity exec --nv ~/singularity_env/nicobert-gpu/nicobert-gpu.sif \
python3 $1