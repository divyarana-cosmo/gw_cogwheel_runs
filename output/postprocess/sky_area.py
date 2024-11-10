from glob import glob
import matplotlib.pyplot as plt
import json 
import numpy as np
import corner
import healpy as hp
import pandas as pd
from os.path import exists
plt.rcParams.update({"text.usetex": False })

import sys


def get_sky_area(ra,dec):
    """infers a rough 90 percent credible region, conservatively"""
    # give in radians we change to degree
    ra      = np.array(ra)*180/np.pi
    dec     = np.array(dec)*180/np.pi
    nside   = 128
    msk     = np.zeros(12*nside**2)
    ipix    = hp.ang2pix(nside, ra, dec, lonlat=1)
    uipix, cnt = np.unique(ipix, return_counts=1)
    msk[uipix] = cnt/len(ra)

    ar = sum(np.cumsum(-1*np.sort(-1*msk))<0.9) * hp.nside2pixarea(nside, degrees=True)
    return ar #area in square degrees


ls_mask = hp.read_map("/home/divya.rana/github/gw_cogwheel_runs/DataStore/lrgs_decals_lowz/full_decals_fp_msk.fits", dtype=float);


for relno in range(12):
    input_data = pd.read_csv('/home/divya.rana/github/gw_cogwheel_runs/DataStore/o4sim_inputs/params_files_cogwheel/final_inj_params.txt_1y_run_no_kagra_%d'%relno, delim_whitespace=1)

    out = open('o4sim_catalog.dat_1y_run_no_kagra_with_skyarea_%d'%relno,'w')
    out.write("filename\tH\tL\tV\tNetwork_SNR\tmed_ra(rad)\tmed_dec(rad)\tmed_dlum\tdlum_err_p\tdlum_err_m\tprob_med_dlum\tbest_ra(rad)\tbest_dec(rad)\tbest_dlum\tmean_dlum\tstd_dlum\tDecals_ovlp_frac\ttru_ra\ttru_dec(rad)\ttru_dlum\tsky_area\n")
    for runno in range(len(input_data['ra'])):
        fil = '/home/divya.rana/github/gw_cogwheel_runs/output/LVCPrior_uni_dl/o4sim_relno_%d_run_%d/run_*'%(relno, runno)
        flist = len(glob(fil))
        fil = '/home/divya.rana/github/gw_cogwheel_runs/output/LVCPrior_uni_dl/o4sim_relno_%d_run_%d/run_%d'%(relno, runno,flist-1)
        postfile = fil + '/samples.feather'        
        chk = exists(postfile)
        if not chk:
            continue
        msknside = int(np.sqrt(len(ls_mask)/12))
        fp = ls_mask
        idx = (fp<1)
        fp[idx] = hp.UNSEEN
        hp.mollview(hp.ma(fp))
    
        #collecting the input values               
        s1          = input_data['sn1'][runno]
        s2          = input_data['sn2'][runno]
        s3          = input_data['sn3'][runno]
        tru_ra      = input_data['ra'][runno]
        tru_dec     = input_data['dec'][runno]
        tru_dlum    = input_data['d_luminosity'][runno]
        s           = input_data['snr'][runno]
    
        #reading posterior data from postfile
        try:
            data = pd.read_feather(postfile)
        except:
            continue
        dat = np.transpose([data['ra'],data['dec'],data['d_luminosity']])
        
        #extracting just the 3d location part to be used in x-corr measure
        np.savetxt('./plots_1y_run_no_kagra/runno_%d_relno_%d_loc.dat'%(runno, relno), dat)
        bidx = np.argmax(data['lnl'])
        phi = dat[:,0]
        theta = np.pi/2 - dat[:,1]
        ipix = hp.ang2pix(msknside,theta,phi)
        
        flg = np.sum(ls_mask[ipix]>=1)*1.0/len(phi)
        med_ra, med_dec, med_dlum = np.median(dat[:,:], axis=0)
        
        best_ra, best_dec, best_dlum = dat[bidx,:]
        dlum_err_p  = np.percentile(dat[:,2], 84) - med_dlum
        dlum_err_m  = med_dlum - np.percentile(dat[:,2], 16)
        mean_dlum   = np.mean(dat[:,2])
        std_dlum    = np.std(dat[:,2])
    
        from scipy.stats import gaussian_kde
        prob_med_dlum = gaussian_kde(dat[:,2]).pdf(med_dlum)[0]
        hp.projscatter(theta, phi, s=0.5, c='y', label='%s'%fil.split('/')[-1])
        hp.projscatter(np.pi/2 - tru_dec, tru_ra, marker='o', edgecolors='black', facecolors='none', label='true-pos')
        hp.projscatter(np.pi/2 - med_dec, med_ra, marker='x',color='red', label='med-pos')
        hp.projscatter(np.pi/2 - best_dec, best_ra, marker='x',color='blue', label='best-pos')
        plt.title("H=%2.2f, L=%2.2f, V=%2.2f, Network SNR=%2.2f, olpfrac=%f"%(s1,s2,s3,s,flg))
        plt.legend()
        if (np.sort([s1,s2,s3])[1]>5) and (s>10) and (flg>0.8):
            plt.savefig('./plots_1y_run_no_kagra/runno_%d_relno_%d.png'%(runno,relno),dpi=300)
            #plt.savefig('./plots_1y_run_no_kagra/%d.png'%runno,dpi=300)
        plt.clf()
        ra  =   data['ra']
        dec =   data['dec']
        sky_area = get_sky_area(ra,dec)
    
        out.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(fil.split('/')[-1], s1, s2, s3, s, med_ra, med_dec, med_dlum, dlum_err_p, dlum_err_m, prob_med_dlum, best_ra, best_dec, best_dlum, mean_dlum, std_dlum, flg, tru_ra, tru_dec, tru_dlum, sky_area))
    
        print(runno,fil,flg)
    
    out.close()    
