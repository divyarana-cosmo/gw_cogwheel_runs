import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

relno=10
df = pd.read_csv('/home/divya.rana/github/gw_bilby_runs_cit/DataStore/o4sim_inputs/params_files_cogwheel/final_inj_params.txt_1y_run_no_kagra_%s'%relno, sep='\s+')

data = np.loadtxt('resubmit.dat')
rows  = np.arange(len(df['m1']))
relno = data[:,1]//1.0
idx = np.isin(rows, relno)

plt.subplot(2,2,1)
plt.scatter(df['m1'][~idx], df['m2'][~idx], s=1.0, label='total: %d,good: %d'%(len(idx), sum(~idx)))
plt.scatter(df['m1'][idx], df['m2'][idx], s=1.0, label='bad:%d'%sum(idx))
plt.xlabel('m1')
plt.ylabel('m2')
plt.legend()

plt.subplot(2,2,2)
bins = np.linspace(0,2000,30)
plt.hist(df['d_luminosity'][~idx], histtype='step',  bins=bins)
plt.hist(df['d_luminosity'][idx], histtype='step',  bins=bins)
plt.xlabel('Dlum')
plt.yscale('log')

plt.subplot(2,2,3)
#bins = np.linspace(0,200,30)
mchirp =(df['m1']*df['m2'])**(3/5) / (df['m1'] + df['m2'])**(1/5)
q = df['m1']/df['m2']
plt.scatter(q[~idx], df['d_luminosity'][~idx],  s=1.0)
plt.scatter(q[idx],  df['d_luminosity'][idx],   s=1.0)
#plt.hist(q[~idx], histtype='step',  bins=15)
#plt.hist(q[idx], histtype='step',  bins=15)
plt.xlabel('mass ratio')
plt.yscale('log')

plt.subplot(2,2,4)
bins = np.linspace(0,50,30)
plt.hist(df['snr'][~idx], histtype='step',  bins=bins)
plt.hist(df['snr'][idx], histtype='step',  bins=bins)
plt.xlabel('SNR')
plt.yscale('log')

plt.tight_layout()
plt.savefig('test.png', dpi=300)
