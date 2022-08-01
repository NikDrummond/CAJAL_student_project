#%% load stuff

import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.rcParams['font.size'] = 15
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

result = pickle.load(open('results/dim_by_mu_and_var.p', 'rb'))
Ks, ps, dims = [result[k] for k in ['Ks', 'ps', 'dims']]

#%% plot stuff

plt.figure(figsize = (4.5,4))
for ip in range(len(ps)):
  plt.plot(Ks, dims[ip, :, 0]/N, '-', color = 'k', alpha = 1 - ip / len(ps) * 0.5, lw = 0.5)
  plt.plot(Ks, dims[ip, :, 1]/N, '-', color = 'r', alpha = 1 - ip / len(ps) * 0.5, lw = 0.5)
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
  ds = dims[:, iK, :]/N
  #ds = ds  / np.mean(ds, axis = 0, keepdims = True)
  #plt.plot(ps, ds[:, 0], '-', color = 'k', alpha = 1 - iK / len(Ks) * 0.5, lw = 0.5)
  plt.plot(ps, ds[:, 1], ls = '-', color = np.ones(3)*(iK/ len(plotis) * 0.7), lw = 2, label = Ks[iK])
plt.xlabel('p')
plt.ylabel('Dimensionality')
plt.title('Mushroom body', pad = 20)
plt.ylim(5.5, 5.7)
plt.legend()
plt.show()