#!/bin/bash

#$-l rt_F=8
#$-l h_rt=72:00:00
#$-l USE_SSH=1
#$-v SSH_PORT=2299
#$-j y 
#$ -cwd

source /etc/profile.d/modules.sh
source /etc/profile.d/uge.sh

module load singularitypro gcc/9.3.0 python/3.8/3.8.7 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 openmpi/4.0.5

export OMP_NUM_THREADS=20
export NGPU_PER_NODE=4 # rt_F:4, rt_AF: 8

FILENAME=train_mlm.py

# launch on slave nodes
node_rank=1
for slave_node in `cat $SGE_JOB_HOSTLIST | awk 'NR != 1 { print }'`; do
qrsh -inherit -V -cwd $slave_node singularity exec --nv ~/singularity_env/nicobert-gpu/nicobert-gpu.sif \
python3 -m torch.distributed.launch --use_env --nproc_per_node $NGPU_PER_NODE --nnodes $NHOSTS --node_rank $node_rank --master_addr `hostname` $FILENAME \
--model ./config/model_config.json \
--data ./config/data_config.json \
--training ./config/training_config_large.json \
--tokenizer ./config/tokenizer_config_large.json &
node_rank=`expr $node_rank + 1`
done

# launch on master node
node_rank=0
singularity exec --nv ~/singularity_env/nicobert-gpu/nicobert-gpu.sif \
python3 -m torch.distributed.launch --use_env --nproc_per_node $NGPU_PER_NODE --nnodes $NHOSTS --node_rank $node_rank --master_addr `hostname` $FILENAME \
--model ./config/model_config.json \
--data ./config/data_config.json \
--training ./config/training_config_large.json \
--tokenizer ./config/tokenizer_config_large.json

# finalize
wait
exit 0
