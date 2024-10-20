#!/bin/bash
#MYJOB_DIR= # Replace here with the directory you are hosting all the scripts
#cd ${MYJOB_DIR}

source /home/divya.rana/.bashrc
#conda activate 
conda activate myenv_py39
mkdir -p log_PE
python /home/divya.rana/github/gw_cogwheel_runs/15d_cbc_fid.py $1 $2


#echo Task $1 finished.
