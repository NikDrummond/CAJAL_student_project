#%% load some stuff
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import get_opt_dim_var, get_anal_rho, get_anal_dim, get_opt_var

plt.rcParams['font.size'] = 15
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

result = pickle.load(open('results/analytical_dims.p', 'rb'))
Ks, ps, incval, dims_K, vars_K, Ms, N, S = [result[k] for k in ['Ks', 'ps', 'incval', 'dims_K', 'vars_K', 'Ms', 'N', 'S']]


#%% plot some stuff

opts, opt_vars = [], []
plt.figure()
for i in range(len(Ks)):
  y = vars_K[i]
  x = ps*(incval**2) #variance of p(K)
  yanal = get_anal_rho(Ks[i], x, N) #analytical result
  print(Ks[i], x[np.argmin(y)], np.amin(y))
  opts.append(x[np.argmin(y)])
  opt_vars.append(np.amin(y))
  plt.scatter(x, y / y[0], label = Ks[i], alpha = 1, s = 20, marker = 'x')
  plt.plot(x, yanal / yanal[0], lw = 1, ls = '-')
plt.xlabel('variance of p(K)')
plt.ylabel('<S_ij^2>')
plt.ylim(0.975, 1.035)
plt.legend(ncol = 3)
plt.show()

opts_dim, opt_vars_dim = [], []
plt.figure()
for i in range(len(Ks)):
  y = dims_K[i]
  x = ps*(incval**2) #variance of p(K)
  yanal = get_anal_dim(Ks[i], x, N, M = Ms[i], approx = False) #analytical result
  plt.scatter(x, y, label = Ks[i], alpha = 1, s = 20, marker = 'x')
  plt.plot(x, yanal, lw = 1, ls = '-')
  opts_dim.append(x[np.argmax(y)])
  opt_vars_dim.append(np.amin(y))
plt.xlabel('variance of p(K)')
plt.ylabel('dim(y)')
plt.show()

plt.figure()
plt.scatter(Ks, opts)
mus = np.linspace(np.amin(Ks)-1, np.amax(Ks)+1, 51)
plt.plot(mus, get_opt_var(mus, N), 'k-')
plt.xlabel('mean K')
plt.ylabel('optimal variance')
plt.show()

plt.figure()
plt.scatter(Ks, opts_dim)
mus = np.linspace(np.amin(Ks)-1, np.amax(Ks)+1, 51)
plt.plot(mus, get_opt_dim_var(mus, S/mus, N), 'k-')
plt.plot(mus, get_opt_var(mus, N), 'k--')
plt.xlabel('mean K')
plt.ylabel('optimal variance (dim)')
plt.show()
# %% plot energy landscapes

mus = np.linspace(1, 16, 101)
vars = np.linspace(0, 14, 91)
S = 8531

all_dims = np.array([get_anal_dim(mus[i], vars, N, M = S/mus[i], approx = False) for i in range(len(mus))]) #mu by var
all_dims[vars[None, :] > mus[:, None]**2] = np.nan
plt.figure()
plt.imshow(all_dims, cmap = 'viridis', aspect = 'auto', vmin = 0.95*np.nanmax(all_dims),
           vmax = np.nanmax(all_dims), extent = (vars[0], vars[-1], mus[-1], mus[0]))
plt.plot(get_opt_dim_var(mus, S/mus, N), mus, 'k-')
plt.xlabel('variance')
plt.ylabel('mean')
plt.colorbar()
plt.show()

print(np.nanmax(all_dims))
print(np.where(all_dims == np.nanmax(all_dims)))

# %%
