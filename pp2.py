import numpy as np
import matplotlib.pyplot as plt
import bilby
import json
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as ius
plt.rcParams.update({"text.usetex": False})
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
def get_pp_cal(quantarr):
    xx = np.linspace(0,1,1001)
    yy = 0.0*xx
    for ii, pp in enumerate(xx):
        yy[ii] = sum(quantarr<pp)/len(quantarr)
    return xx,yy

# need a code to read the data from all the feather filee and truth

def getquant(plist,relno=10):
    from glob import glob
    relno = relno
    df = pd.read_csv('/home/divya.rana/github/gw_bilby_runs_cit/DataStore/o4sim_inputs/params_files_cogwheel/final_inj_params.txt_1y_run_no_kagra_%s'%relno, sep='\s+')
    print("read the truth file")
    #plist = df.keys()
    plist = plist
    quantarr = {}
    for pp in plist:
        quantarr[pp] = np.array([])
    
    Nevents = len(df.values[:,0])
    
    quantarr['runno'] = np.array([])
    
    for runno in range(Nevents):
        #SNR cut 
        if df['snr'][runno]<12:
            continue
    
        flist = glob('output/LVCPrior_uni_dl/o4sim_relno_%d_run_%d/run_*/samples.feather'%(relno, runno))
        if len(flist)==0:
            continue
        else:
            fil = 'output/LVCPrior_uni_dl/o4sim_relno_%d_run_%d/run_%d/samples.feather'%(relno, runno, len(flist)-1)
        try:    
            samples = pd.read_feather(fil)
        except:
            continue
        
        for pp in plist:
            #print(pp)
            truth     = df[pp][runno]    
            quantarr[pp] = np.append(quantarr[pp], get_quant(truth, samples[pp].values))
        
        quantarr['runno']    =   np.append(quantarr['runno'],runno)    
        print(runno)
    
    #Save the quantiles in a file
    df = pd.DataFrame(quantarr)
    df.to_csv('output/LVCPrior_uni_dl/quantiles_relno_%d.dat'%(relno), index=False)
    return 0

#relno =0
#for relno in range(19,20):
#    plist = ['ra', 'dec', 'd_luminosity', 'iota']
#    getquant(plist, relno=relno)
## read from my file
#quantarr = pd.read_csv('output/LVCPrior_uni_dl/quantiles_relno_%d.dat'%(relno))
#
#make plots
plist = ['ra', 'dec', 'd_luminosity', 'iota']
llist = [r'$ra$', r'$dec$', r'$d_{\rm lum}$', r'$\iota$']
bins = np.linspace(0,1,101)
for cnt,pp in enumerate(plist[:-1]):
    plt.subplot(3,3,cnt+1)
    for relno in range(20):
        quantarr = pd.read_csv('output/LVCPrior_uni_dl/quantiles_relno_%d.dat'%(relno))
        xx,yy = get_pp_cal(quantarr[pp])
        plt.plot(xx,yy)
    print(pp)
    plt.plot(xx,xx,'--k')
    plt.xlim(0,1.0)
    plt.ylim(0,1.0)
    plt.title(llist[cnt])    
    plt.xlabel(r'Credible interval')
    if cnt==0:
        plt.ylabel(r'Fraction of injections')

plt.tight_layout()
plt.savefig('pp_plots.png', dpi=300)
#plt.savefig('test.png', dpi=300, bbox_inches='tight')

#plt.hist(quantarr[pp], histtype='step', bins=bins)
#plt.title(pp)    
#plt.legend()
#plt.xlabel(r'Credible interval')
#plt.legend(fontsize='small')
#plt.savefig('test.png', dpi=300, bbox_inches='tight')
#
#
#plt.clf()
#
#for cnt,pp in enumerate(plist):
#    xx,yy = get_pp_cal(quantarr[pp])
#    plt.plot(xx,yy, label = pp)
#
#plt.plot(xx,xx,'--k')
#plt.xlim(0,1.0)
#plt.ylim(0,1.0)
#plt.xlabel(r'Credible interval')
#plt.ylabel(r'Fraction of injections in credible interval')
#plt.legend(fontsize='small')
#plt.savefig('test1.png', dpi=300, bbox_inches='tight')

