#%%use drosophila J

from toolbox import models
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from params import params
import sys

#%%
J_G_o = np.load('niks_crap/glom_J.npy')
J_P_o = np.load('niks_crap/PN_J.npy')
Ks_G, Ks_P = np.sum(J_G_o, axis = 0).astype(int), np.sum(J_P_o, axis = 0).astype(int)
Gs_G, Gs_P = np.sum(J_G_o, axis = 1).astype(int), np.sum(J_P_o, axis = 1).astype(int)
J_G, J_P = J_G_o - np.mean(J_G_o), J_P_o - np.mean(J_P_o)

N, M = J_G.shape
T = 20000

#%%
def get_lin_nonlin(J, Print = False):
    Clin = J.T @ J
    lin_dim = np.trace(Clin)**2 / np.sum(Clin**2) / N

    X = np.random.normal(0, 1, (T,N))
    R = models.run_model(params, J, X)
    nonlin_dim = models.get_dimM(R) / N

    if Print: print(lin_dim, nonlin_dim)
    return lin_dim, nonlin_dim

lin_dim, nonlin_dim = get_lin_nonlin(J_G, Print = True)

# %%

Nrep = 200
lins, nonlins = np.zeros((Nrep, 2)), np.zeros((Nrep, 2))
for n in range(Nrep):
    if n % 10 == 1: print(n, np.mean(nonlins[:n, :], axis = 0))
    newJ = np.zeros(J_G.shape)
    for i in range(M):
        newJ[np.random.choice(N, Ks_G[i], replace = False), i] = 1
    assert np.sum(newJ) == np.sum(J_G_o)
    newJ -= np.mean(newJ)
    newlin, newnonlin = get_lin_nonlin(newJ)
    lins[n, 0] = newlin
    nonlins[n, 0] = newnonlin

    newJ = np.zeros(J_G.shape)
    for i in range(N):
        newJ[i, np.random.choice(M, Gs_G[i], replace = False)] = 1
    assert np.sum(newJ) == np.sum(J_G_o)
    newJ -= np.mean(newJ)
    newlin, newnonlin = get_lin_nonlin(newJ)
    lins[n, 1] = newlin
    nonlins[n, 1] = newnonlin

result = {'lin_dim': lin_dim, 'nonlin_dim': nonlin_dim, 'lins': lins, 'nonlins': nonlins}
pickle.dump(result, open('./results/drosophila_J_dim.p', 'wb'))




# %%
