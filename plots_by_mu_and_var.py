#%% load stuff

import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import get_opt_dim_var

plt.rcParams['font.size'] = 15
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

result = pickle.load(open('results/dim_by_mu_and_var.p', 'rb'))
Ks, ps, dims, params = [result[k] for k in ['Ks', 'ps', 'dims', 'params']]
N, S, incval = [params[k] for k in ['N', 'S', 'stepK']]
#%% plot stuff

plt.figure(figsize = (4.5,4))
for ip in range(len(ps)):
  plt.plot(Ks, dims[:, ip, 0, 0], '-', color = 'k', alpha = 1 - ip / len(ps) * 0.5, lw = 0.5)
  plt.plot(Ks, dims[:, ip, 1, 0], '-', color = 'r', alpha = 1 - ip / len(ps) * 0.5, lw = 0.5)
plt.xlabel('K')
plt.ylabel('Dimensionality/N')
plt.ylim(0, np.amax(dims)/N*1.1)
plt.title('Mushroom body', pad = 20)
plt.xticks([0,5,10,15,20])
plt.xlim(0, 21)
plt.yticks(range(7))
plt.ylim(0, 6)
plt.show()

plt.figure()
plotis = range(2,6)
for iK in plotis:
  print(Ks[iK])
  ds = dims[iK, :, 1, 0]
  plt.plot(ps, ds, ls = '-', color = np.ones(3)*(iK/ len(plotis) * 0.7), lw = 2, label = Ks[iK])
plt.xlabel('p')
plt.ylabel('Dimensionality')
plt.title('Mushroom body', pad = 20)
plt.ylim(5.0,5.2)
#plt.ylim(5.5, 5.7)
plt.legend()
plt.show()

#%% plot some more stuff

J_G = np.load('niks_crap/glom_J.npy')
Ks_G = np.sum(J_G, axis = 0)

vars = ps*incval**2
mus = np.array(Ks)
dv = (vars[1] - vars[0])/2

norm_dims = (dims[..., 1, 0] / dims[:, :1, 1, 0]) #mean by var
nonorm = dims[..., 1, 0]
print(norm_dims.shape)
maxvars = vars[np.argmax(norm_dims, axis = 1)]

plt.figure()
plt.imshow(norm_dims, cmap = 'viridis', aspect = 'auto', vmin = 0.97, vmax = 1.015, extent = (vars[0]-dv, vars[-1]+dv, len(mus)-0.5, -0.5))
plt.colorbar()
plt.scatter(maxvars, np.arange(len(mus)), color = 'k')
plt.plot(get_opt_dim_var(mus, S/mus, N), np.arange(len(mus)), 'k-')
plt.yticks(np.arange(len(mus))[::2], mus[::2])
plt.xlabel('variance')
plt.ylabel('mean')
plt.show()

plt.figure()
plt.imshow(nonorm, cmap = 'viridis', aspect = 'auto', extent = (vars[0]-dv, vars[-1]+dv, len(mus)-0.5, -0.5),
            vmin = 0.95*np.nanmax(nonorm), vmax = np.nanmax(nonorm))
plt.colorbar()
plt.scatter([np.var(Ks_G)], [np.mean(Ks_G)- mus[0]], color = 'r', marker = 'x', s = 100)
plt.yticks(np.arange(len(mus))[::2], mus[::2])
plt.xlabel('variance')
plt.ylabel('mean')
plt.show()
# %%
