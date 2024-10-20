import numpy as np
from glob import glob
from os.path import exists
from subprocess import call
import pandas as pd

fout = open('resubmit.dat', 'w')
#fout.write('#relno\trunno\n')
#for relno in range(10,20):A
np.random.seed(123)
for relno in range(10,11):
    data = pd.read_csv('./DataStore/o4sim_inputs/params_files_cogwheel/final_inj_params.txt_1y_run_no_kagra_%d'%relno, delim_whitespace=1)
    for runno in range(len(data['ra'])):
        fil = 'output/LVCPrior/o4sim_relno_%d_run_%d/run_*'%(relno, runno)
        flist = len(glob(fil))
        fil = 'output/LVCPrior/o4sim_relno_%d_run_%d/run_%d'%(relno, runno,flist-1)
        jfil = fil + '/samples.feather'        
        chk = exists(jfil)
        if not chk:
            fout.write('%d\t%d\n'%(relno,runno))
            call('rm -r output/LVCPrior/o4sim_relno_%d_run_%d'%(relno, runno), shell=1)
            #call("mv %s ~/github/gw_bilby_runs_cits/output/" % fil, shell=1)
        print(fil)

fout.close()



 
#flist = glob('output/o4sim_*')
#fout = open('resubmit.dat', 'w')
#fout.write('#relno\trunno\n')
#print(len(flist))
#for fil in flist:
#    relno = int(fil.split('_')[2])
#    runno = int(fil.split('_')[4])

