import numpy as np
import matplotlib.pyplot as plt
import pickle

#%% define a function for sampling weights 
def get_J_var(K, N, M, rep, p = 0.2, inc = 1):
    assert K-inc > -0.5
    J = np.zeros((rep, N, M)) #weight matrix

    Kvals = [K-inc, K+inc, K]
    pvals = np.array([p/2, p, np.inf])
    for r in range(rep):
        for i in range(M):
            Ki = Kvals[int(np.sum(np.random.uniform() > pvals))]
            J[r, np.random.choice(N, Ki, replace=False), i] = 1
    return J

def get_dim(C):
    trs = np.trace(C, axis1=1, axis2=2)
    dim = trs**2/np.sum(C**2, axis = (1,2))
    return np.mean(dim)


#%% run some simulations

S = 14000
rep = 20
N = 50

Ks = np.array([5,6,8,10,12,14,16,18])
Ms = np.round(S/Ks).astype(int)
ps = np.linspace(0, 1.0, 21)
vars_K = []
dims_K = []
incval = 4
for iK, K in enumerate(Ks):
  vars_K.append([])
  dims_K.append([])
  for ip, p in enumerate(ps):
    J = get_J_var(K, N, Ms[iK], rep, p = p, inc = incval)
    J -= np.mean(J)
    C = J.transpose(0, 2, 1) @ J
    dims_K[-1].append(get_dim(C))
    C[:, np.arange(C.shape[1]), np.arange(C.shape[2])] = np.nan
    if K % 4 == 2 and ip % 3 == 0:
      print(K, p, np.nanmean(C), np.nanvar(C), dims_K[-1][-1])
    vars_K[-1].append(np.nanvar(C))

result = {'Ks': Ks, 'ps': ps, 'dims_K': dims_K, 'vars_K': vars_K, 'incval': incval}
pickle.dump(result, open('results/analytical_dims.p', 'wb'))

