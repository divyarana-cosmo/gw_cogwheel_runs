
# Ensure only one core is used
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
lal.swig_redirect_standard_output_error(False)

from cogwheel import posterior, data, sampling, gw_plotting



# ## Directory setup
# Edit these as desired:



parentdir = 'example'  # Directory that will contain parameter estimation runs
eventname = 'my_inj'


# Create an injection

# Instantiate synthetic Gaussian noise with duration, ASD functions and detector names
event_data = data.EventData.gaussian_noise(
    eventname=eventname, duration=8, detector_names='HLV',
    asd_funcs=['asd_H_O4', 'asd_L_O4', 'asd_V_O4'], tgps=0.0)

# Inject a signal on top
par_dic = {'m1': 2.0,
           'm2': 2.0,
           'l1': 0,
           'l2': 0,
           'd_luminosity': 100.0,
           'iota': np.pi / 4,
           'phi_ref': np.pi / 5,
           'ra': 2.4,
           'dec': 0.15,
           'psi': 0.5,
           's1z': 0.0,
           's2z': 0.0,
           's1x_n': 0.0,
           's1y_n': 0.0,
           's2x_n': 0.0,
           's2y_n': 0.0,
           't_geocenter': 0.0,
           'f_ref': 105.0}

event_data.inject_signal(par_dic=par_dic, approximant='IMRPhenomXAS')

## Plot spectrogram
#event_data.specgram((-0.1, 0.1))


# Run parameter estimation

# Maximize likelihood, set up relative-binning summary data:\n
post = posterior.Posterior.from_event(event=event_data, mchirp_guess=None, approximant='IMRPhenomXAS', prior_class='IntrinsicAlignedSpinIASPrior', prior_kwargs={'symmetrize_lnq': True, 'f_ref': par_dic['f_ref']})


# Run the sampler
pym = sampling.PyMultiNest(post)
pym.run_kwargs['n_live_points'] = 256
rundir = pym.get_rundir(parentdir)
pym.run(rundir)


# ### Plot posteriors

# In[11]:


par_dic.update(post.prior.inverse_transform(**par_dic))

# Load samples
samples = pd.read_feather(rundir/'samples.feather')


# In[12]:


plot_params = ['mchirp', 'lnq', 'chieff', 'd_luminosity', 'iota', 
               'ra', 'dec', 'psi', 'phi_ref']

cp = gw_plotting.CornerPlot(samples, params=plot_params)

cp.plot(max_n_ticks=3)
cp.scatter_points(par_dic, colors=['C3'], adjust_lims=True)


# In[ ]:




