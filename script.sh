#!/bin/bash
##!/bin/bash

# export https_proxy="http://proxy.ftm.alcf.anl.gov:3128"
# export http_proxy="http://proxy.ftm.alcf.anl.gov:3128"
# export ftp_proxy="http://proxy.ftm.alcf.anl.gov:3128"


##-------------------------------------------------------
### The following is for running on JLSE
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate jax__
##-------------------------------------------------------


# #-------------------------------------
# # Sine
# python run.py train 1 "param2.json"
# ##--------------------------------------
# ## Graph Synthetic
# python run.py train 1 "paramgraph0.json"
##--------------------------------------
## Omni
python run.py train 1 "paramomni9.json"
##--------------------------------------
## MNIST
python run.py train 1 "paramomni10.json"



##--------------------------------------
# # Graph ENZYMES
# python run.py train 1 "paramgraph1.json"
##--------------------------------------
# # Graph MUTAG
# python run.py train 5 "paramgraph2.json" 
# #--------------------------------------