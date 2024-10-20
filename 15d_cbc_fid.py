
# Ensure only one core is used
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
lal.swig_redirect_standard_output_error(False)
import lalsimulation as lalsim
from cogwheel import posterior, data, sampling, gw_plotting, likelihood, gw_utils
import matplotlib.pyplot as plt

import bilby


def run_pe(relno, runno):
    # read the parameters from the file
    df = pd.read_csv('/home/divya.rana/github/gw_bilby_runs_cit/DataStore/o4sim_inputs/params_files_cogwheel/final_inj_params.txt_1y_run_no_kagra_%s'%relno, sep='\s+')

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
    
    #sampling_frequency = 2048.
    parentdir = 'output'
    from subprocess import call
    call("mkdir -p %s"%parentdir, shell=1)
    eventname ='o4sim_relno_%d_run_%d'%(relno, runno) 
 
    # inject them into the right noise
    event_data = data.EventData.gaussian_noise(eventname=eventname, duration=duration, detector_names='HLV',asd_funcs=['asd_H_O4', 'asd_L_O4', 'asd_V_O4'], tgps=tgps, seed=seed)
    event_data.inject_signal(par_dic=par_dic, approximant='IMRPhenomXPHM')
    #print(par_dic, tgps, df['snr'][runno])
    # run the pe with IASprior
    post = posterior.Posterior.from_event(event=event_data, mchirp_guess=gw_utils.m1m2_to_mchirp(par_dic['m1'], par_dic['m2']), approximant='IMRPhenomXPHM', prior_class='LVCPrior', prior_kwargs={'f_ref': f_ref}, ref_wf_finder_kwargs={'time_range': (-1.0,1.0)} )
    #post = posterior.Posterior.from_event(event=event_data, mchirp_guess=gw_utils.m1m2_to_mchirp(par_dic['m1'], par_dic['m2']), approximant='IMRPhenomXPHM', prior_class='LVCPrior', prior_kwargs={'f_ref': f_ref} )
    #post = posterior.Posterior.from_event(event=event_data, mchirp_guess=gw_utils.m1m2_to_mchirp(par_dic['m1'], par_dic['m2']), approximant='IMRPhenomXPHM', prior_class='LVCPrior', prior_kwargs={'f_ref': f_ref, 'd_luminosity_max': 10000} )




    # Run the sampler
    #pym = sampling.Dynesty(post)
    pym = sampling.PyMultiNest(post)
    pym.run_kwargs['n_live_points'] = 512
    #pym.run_kwargs['n_live_points'] = 256
    
    rundir = pym.get_rundir(parentdir)
    pym.run(rundir)

    # some plotting
    plot_params = ['mchirp', 'lnq', 'd_luminosity', 'iota', 'ra', 'dec', 'psi',            'phi_ref']
    #plot_params = ['mchirp', 'lnq', 'chieff', 'd_luminosity', 'iota', 'ra', 'dec',        'psi', 'phi_ref']
    samples = pd.read_feather(rundir/sampling.SAMPLES_FILENAME)
    csp = gw_plotting.CornerPlot(samples, params=plot_params)
    csp.plot(max_n_ticks=3)
    csp.scatter_points(par_dic, colors=['C3'], adjust_lims=True)
    plt.savefig(rundir/f'{eventname}.pdf', bbox_inches='tight')
    return 0

if __name__ == "__main__":
    import sys
    relno = int(sys.argv[1])
    runno = int(sys.argv[2])
    run_pe(relno, runno)
