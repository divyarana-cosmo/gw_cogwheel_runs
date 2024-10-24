import numpy as np
import matplotlib.pyplot as plt
import bilby
import json
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as ius

# needs some touch up for using with feather  file


# get the cumulative distribution from the weights & samples and put the spline
def samp2cumsum_spl(samples, weights):
    idx = np.argsort(samples)
    samples = samples[idx]
    weights = weights[idx]
    cumweight = np.cumsum(weights)/sum(weights)
    return ius(samples,cumweights, ext=3)#if it is outside then 0 or 1

# use the spline to get the quantile of the true parameter values
def get_quant(truth, samples, weights):
    value  = samp2cumsum_spl(samples, weights)(truth)
    return value

# then to the p-p plot 
def get_pp_cal(quantarr, binsize=0.1):
    idx = quantarr//binsize
    uqidx, counts = np.unique(idx, return_counts=1)
    prob = np.cumsum(counts)/len(quantarr)
    return uqidx*binsize, prob

# need a code to read the data from all the feather filee and truth

from glob import glob
relno = 10
df = pd.read_csv('/home/divya.rana/github/gw_bilby_runs_cit/DataStore/o4sim_inputs/params_files_cogwheel/final_inj_params.txt_1y_run_no_kagra_%s'%relno, sep='\s+')

plist = df.keys()
plist = ['ra', 'dec', 'd_luminosity']

Nevents = len(df.values[:,0])
quantarr = -999*np.ones(Nevents)
flist = glob('output/LVCPrior/o4sim_relno_%d_run_%d/run_*/samples.feather')%(relno, runno)

for runno in range(Nevents):
    flist = glob('output/LVCPrior/o4sim_relno_%d_run_%d/run_*/samples.feather')%(relno, runno)
    if len(flist)==0:
        continue
    else:
        fil = 'output/LVCPrior/o4sim_relno_%d_run_%d/run_%d/samples.feather'%(relno, runno, len(flist)-1)
        
    samples = pd.read_feather(fil)

    



par_dic = { 'm1': df['m1']          [runno]
 ,'m2'          : df['m2']          [runno]
 ,'l1'          : df['l1']          [runno]  
 ,'l2'          : df['l2']          [runno] 
 ,'d_luminosity': df['d_luminosity'][runno] 
 ,'iota'        : df['iota']        [runno] 
 ,'phi_ref'     : df['phi_ref']     [runno] 
 ,'ra'          : df['ra']          [runno]
 ,'dec'         : df['dec']         [runno] 
 ,'psi'         : df['psi']         [runno]
 ,'s1z'         : df['s1z']         [runno]
 ,'s2z'         : df['s2z']         [runno]
 ,'s1x_n'       : df['s1x_n']       [runno]
 ,'s1y_n'       : df['s1y_n']       [runno]
 ,'s2x_n'       : df['s2x_n']       [runno]
 ,'s2y_n'       : df['s2y_n']       [runno]
 ,'t_geocenter' : df['t_geocenter'] [runno]
 ,'f_ref'       : df['f_ref']       [runno]}
#print(par_dic)
f_ref       = df['f_ref'][runno] 
seed        = df['seed'][runno]
tgps        = df['t_geocenter'][runno] - 2
duration    = 4
 
flist = glob('output/LVCPrior/o4sim_relno_%d_run_%d/run_*/samples.feather')%(relno, runno)
flist = len(glob(fil))
fil = 'output/LVCPrior/o4sim_relno_%d_run_%d/run_%d'%(relno, runno,flist-1)
jfil = fil + '/samples.feather'   



#xbin = np.linspace(0,1,10) #p_obs
#rbin = np.zeros(len(xbin)) #p_mes
import healpy as hp

mask = hp.read_map("../lrgs_decals_lowz/full_decals_fp_msk.fits", dtype=float);
msknside = int(np.sqrt(len(mask)/12))

from glob import glob
flist = np.sort(glob("./output_1y_run_no_kagra/run_*_result.json"))

lb=open('./pp_frac_output_1y_run_no_kagra','w')
lb.write('#frac\n')
cnt=0

bp=open('./bad_post_output_1y_run_no_kagra.txt','w')
for i1, fil in enumerate(flist):
    try:
        with open(fil,'r') as f:
            data = json.load(f)

        s1 = data['meta_data']['likelihood']['interferometers']['H1']['optimal_SNR']
        s2 = data['meta_data']['likelihood']['interferometers']['L1']['optimal_SNR']
        s3 = data['meta_data']['likelihood']['interferometers']['V1']['optimal_SNR']
        #s4 = data['meta_data']['likelihood']['interferometers']['K1']['optimal_SNR']
        tru_ra   = data['meta_data']['likelihood']['interferometers']['H1']['parameters']['ra']
        tru_dec  = data['meta_data']['likelihood']['interferometers']['H1']['parameters']['dec']
        tru_dlum = data['meta_data']['likelihood']['interferometers']['H1']['parameters']['luminosity_distance']

        s = np.sqrt(s1**2 + s2**2 + s3**2)

        dat = np.transpose([data['posterior']['content']['ra'],data['posterior']['content']['dec'],data['posterior']['content']['luminosity_distance']])
        phi     = dat[:,0]
        theta   = np.pi/2 - dat[:,1]
        ipix    = hp.ang2pix(msknside,theta,phi)
        flg     = np.sum(mask[ipix]>=1)*1.0/len(phi)

        #if (np.sort([s1, s2, s3, s4])[1] > 5) and (s > 10) and (flg > 0.8):
        if (s > 10) and (flg > 0.8):
            dis  = np.sort(dat[:,2])
            idx  = (dis < tru_dlum)
            frac = len(dis[idx])*1.0/len(dis)
            lb.write('%s\n' % frac)
            print(i1, tru_dlum,frac)
    except:
        print(i1,fil)
        bp.write('%d\t%s\n' % (i1,fil))
lb.close()
bp.close()

