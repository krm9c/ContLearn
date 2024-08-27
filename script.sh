#!/bin/bash
##!/bin/bash

export https_proxy="http://proxy.ftm.alcf.anl.gov:3128"
export http_proxy="http://proxy.ftm.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.ftm.alcf.anl.gov:3128"


############################################################
## THIS IS THE SCRIPT WITH NOISE AND THE VARIANCE CORRECTION.
############################################################

## The following is for running on theta gpu
# export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
# export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh" ]; then
#         . "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# conda activate posei

#-------------------------------------------------------
## The following is for running on JLSE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax__


#--------------------------------------
# Sine
python run.py train 1 "param2.json" 
#--------------------------------------
# Omni
python run.py train 1 "paramomni9.json" 
#--------------------------------------
# Graph Synthetic
python run.py train 1 "paramgraph0.json" 


# # Graph ENZYMES
# python run.py train 1 "paramgraph1.json" 
# # Graph MUTAG
# python run.py train 5 "paramgraph2.json" 
# #--------------------------------------

# python run.py train 10 "param4.json" 
# python run.py train 10 "param6.json" 
# python run.py train 10 "param8.json" 


# python run.py train 10 "param1.json" 
# python run.py train 10 "param3.json" 
# python run.py train 10 "param5.json" 
# python run.py train 10 "param7.json" 


# python run.py train 10 "paramomni9.json" 
# python run.py train 10 "paramomni10.json" 