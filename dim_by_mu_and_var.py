#%% load some stuff
# 
from toolbox import models
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from params import params
import sys

#%% run some comps

params['reps'] = 200
params['stepK'] = 3
distribution = 'step'
Ks = np.arange(3,13)
ps = np.linspace(0, 1, 51)

dims = np.zeros((len(Ks), len(ps), 2, 2))

for ik, K in enumerate(Ks):
    params['K'] = K
    for ip, pval in enumerate(ps):
        params['stepP'] = pval
        dim_all = models.simulate(params, distribution)
        dims[ik,ip,:,0] = np.mean(dim_all, axis=0)
        dims[ik,ip,:,1] = np.std(dim_all, axis=0)
        if ik % 3 == 0 and ip % 3 == 0:
            print('new K', K, pval, dims[ik, ip, :, 0])
            sys.stdout.flush()

#%% save data
result = {'Ks': Ks, 'ps': ps, 'dims': dims, 'params': params}
pickle.dump(result, open('results/dim_by_mu_and_var.p', 'wb'))

#%% plot some stuff
cols = ['k', 'r']
labels = ['no inhibition', 'inhibition']
plt.figure()
for i in range(2):
    m, s = dims[:, 0, i, 0], dims[:, 0, i, 1]/np.sqrt(params['reps'])
    plt.plot(Ks, m, cols[i]+'.-', label=labels[i])
    plt.fill_between(Ks, m-s, m+s, color = cols[i], alpha = 0.2)
plt.legend()
plt.xlabel('K')
plt.ylabel('dim(m)/N')
plt.show()
# %%
