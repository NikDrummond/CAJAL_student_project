import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import get_opt_dim_var, get_anal_rho

plt.rcParams['font.size'] = 15
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

result = pickle.load(open('results/analytical_dims.p', 'rb'))
Ks, ps, incval, dims_K, vars_K = [result[k] for k in ['Ks', 'ps', 'incval', 'dims_K', 'vars_K']]
opts, opt_vars = [], []
plt.figure()
for i in range(len(Ks)):
  y = vars_K[i]
  x = ps*(incval**2) #variance of p(K)
  yanal = get_anal_rho(Ks[i], x) #analytical result
  print(Ks[i], x[np.argmin(y)], np.amin(y))
  opts.append(x[np.argmin(y)])
  opt_vars.append(np.amin(y))
  plt.plot(x, y / y[0], label = Ks[i], alpha = 1)
  plt.plot(x, yanal / yanal[0], color = 'k', lw = 1, ls = '--')
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
  yanal = get_anal_dim(Ks[i], x, M = Ms[i], approx = False) #analytical result
  plt.plot(x, y, label = Ks[i], alpha = 1)
  plt.plot(x, yanal, color = 'k', lw = 1, ls = '--')
  opts_dim.append(x[np.argmax(y)])
  opt_vars_dim.append(np.amin(y))
plt.xlabel('variance of p(K)')
plt.ylabel('dim(y)')
plt.show()

plt.figure()
plt.scatter(Ks, opts)
mus = np.linspace(np.amin(Ks)-1, np.amax(Ks)+1, 51)
plt.plot(mus, get_opt_var(mus), 'k-')
plt.xlabel('mean K')
plt.ylabel('optimal variance')
plt.show()

plt.figure()
plt.scatter(Ks, opts_dim)
mus = np.linspace(np.amin(Ks)-1, np.amax(Ks)+1, 51)
plt.plot(mus, get_opt_dim_var(mus, S/mus), 'k-')
plt.plot(mus, get_opt_var(mus), 'k--')
plt.xlabel('mean K')
plt.ylabel('optimal variance (dim)')
plt.show()