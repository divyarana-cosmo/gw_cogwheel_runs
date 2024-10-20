#!/bin/bash

#setting up the conda environment
conda env create -f bilby_environment.yml

#copy the asd and the changes in the detector file

scp divya.rana@ligo.gravity.cf.ac.uk:/home/divya.rana/.conda/envs/myenv_py37/lib/python3.7/site-packages/bilby/gw/detector/detectors/V2.interferometer /home/divya.rana/.conda/envs/myenv_py37/lib/python3.7/site-packages/bilby/gw/detector/detectors/V1.interferometer

scp -r divya.rana@ligo.gravity.cf.ac.uk:/home/divya.rana/.conda/envs/myenv_py37/lib/python3.7/site-packages/bilby/gw/detector/noise_curves/o4_sim_noise_curves /home/divya.rana/.conda/envs/myenv_py37/lib/python3.7/site-packages/bilby/gw/detector/noise_curves/
