#%%
# 
import numpy as np
import matplotlib.pyplot as plt
from toolbox import models
from params import params

params['T'] = 2000
N, T = params['N'], params['T']
params['corr_ell'] = 3
params["stepP"] =  0.3
rng = np.random.default_rng()

X = rng.normal(0, 1, (T, N))
J = models.get_J(params, rng, 'step', True)

print(np.mean(J))
print(np.mean( J > 0))

# %%

dim = models.get_dimM(J)
print(dim)

dims_Y = []
dims_J = []
dims_O = []
ells = np.linspace(0, 5, 51)
for iell, ell in enumerate(ells+1e-6):
    K = np.exp(-(np.arange(N)[:, None] - np.arange(N)[None, :])**2 / (2*ell**2))
    L = np.linalg.cholesky(K+np.eye(N)*1e-9)
    newJ = np.linalg.inv(L.T)@J

    #qval = np.quantile(newJ, 1-np.sum(J > 0)/newJ.size)
    #newJ = (newJ > qval).astype(float)
    #newJ -= np.mean(newJ)

    dim = models.get_dimM(X @ newJ)
    dims_J.append(dim)
    #print(ell, models.get_dimM(np.linalg.inv(L.T)@J))
    if iell % 10 == 0: print(ell, dim, np.sum(newJ > 0.5))

    Y = X @ L.T @ newJ
    dims_Y.append(models.get_dimM(Y))
    dims_O.append(models.get_dimM(X @ L.T @ J))

#%%

plt.figure()
plt.plot(ells, dims_O, 'b-', label = 'dim(J)')
plt.plot(ells, dims_Y, 'r-', label = 'dim(J^)')
plt.plot(ells, dims_J, 'k-', label = 'dim(J^) uncorr')
plt.legend()
plt.xlabel('correlation level')
plt.ylabel('dimensionality')
plt.show()

# %%
