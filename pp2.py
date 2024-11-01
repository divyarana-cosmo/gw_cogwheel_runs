import numpy as np
import matplotlib.pyplot as plt
import bilby
import json
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as ius

# get the cumulative distribution from the weights & samples and put the spline
def samp2cumsum_spl(samples, weights=None):
    if type(weights)==type(None):
        weights= 1.0 + 0.0*samples
    samples, idx, cnt = np.unique(samples, return_index=True, return_counts=True)
    weights = cnt*weights[idx]
    cumweights = np.cumsum(weights)/sum(weights)
    return ius(samples,cumweights, ext=3)#if it is outside then 0 or 1

# use the spline to get the quantile of the true parameter values
def get_quant(truth, samples, weights=None):
    if type(weights)==type(None):
        return sum(samples<truth)/len(samples)
    value  = samp2cumsum_spl(samples, weights)(truth)
    return value

# then to the p-p plot 
def get_pp_cal(quantarr, binsize=0.05):
    idx = quantarr//binsize
    uqidx, counts = np.unique(idx, return_counts=1)
    prob = np.cumsum(counts)/len(quantarr)
    return (uqidx + 0.5)*binsize, prob

# need a code to read the data from all the feather filee and truth
from glob import glob
relno = 10
df = pd.read_csv('/home/divya.rana/github/gw_bilby_runs_cit/DataStore/o4sim_inputs/params_files_cogwheel/final_inj_params.txt_1y_run_no_kagra_%s'%relno, sep='\s+')
print("read the truth file")
plist = df.keys()
plist = ['ra', 'dec', 'd_luminosity', 'iota']
quantarr = {}
for pp in plist:
    quantarr[pp] = np.array([])

Nevents = len(df.values[:,0])

for runno in range(Nevents):
    #SNR cut 
    if df['snr'][runno]<12:
        continue

    flist = glob('output/LVCPrior_uni_dl/o4sim_relno_%d_run_%d/run_*/samples.feather'%(relno, runno))
    if len(flist)==0:
        continue
    else:
        fil = 'output/LVCPrior_uni_dl/o4sim_relno_%d_run_%d/run_%d/samples.feather'%(relno, runno, len(flist)-1)
    samples = pd.read_feather(fil)
    
    for pp in plist:
        truth     = df[pp][runno]    
        quantarr[pp] = np.append(quantarr[pp], get_quant(truth, samples[pp].values))

       
    print(runno)

#make plots
plt.subplot(2,2,1)
for cnt,pp in enumerate(plist):
    xx,yy = get_pp_cal(quantarr[pp], binsize=0.05)
    plt.plot(xx,yy, label = pp)


plt.plot(xx,xx,'--k')
plt.xlabel(r'$p_{\rm exp}$')
plt.ylabel(r'$p_{\rm obs}$')
plt.legend()
plt.savefig('test.png', dpi=300, bbox_inches='tight')

