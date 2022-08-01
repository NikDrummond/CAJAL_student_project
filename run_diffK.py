#%% load some stuff
# 
from toolbox import models
import json
import numpy as np
import matplotlib.pyplot as plt
from params import params

#%% run some comps

distribution = 'step'
params['stepP'] = 0
Ks = np.arange(2, 10, 1)
dimKs = np.zeros((len(Ks), 2, 2))
params['reps'] = 3
for ik, K in enumerate(Ks):
    print('new K', K)
    params['K'] = K
    dim_all = models.simulate(params, distribution)
    dimKs[ik,:,0] = np.mean(dim_all, axis=0)
    dimKs[ik,:,1] = np.std(dim_all, axis=0)

#%% plot some stuff

plt.figure()
plt.plot(Ks, dimKs[:,0,0], 'k.-', label='No inhibition')
plt.plot(Ks, dimKs[:,1,0], 'r.-', label='Inhibition')
plt.legend()
plt.xlabel('K')
plt.ylabel('dim(m)/N')
plt.show()


# %%
