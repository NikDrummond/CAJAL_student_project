#%% load some stuff
import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.rcParams['font.size'] = 15
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

result = pickle.load(open('./results/drosophila_J_dim.p', 'rb'))
lin_dim, nonlin_dim, lins, nonlins = [result[k] for k in ['lin_dim', 'nonlin_dim', 'lins', 'nonlins']]

# %%
for dat in [(nonlins, nonlin_dim), (lins, lin_dim)]:
    plt.figure()
    plt.hist(dat[0], bins = 10)
    plt.axvline(dat[1], color = 'k')
    plt.show()

# %%
