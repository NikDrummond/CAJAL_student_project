#%% load some stuff
import numpy as np
import matplotlib.pyplot as plt
import pickle
from toolbox import models

plt.rcParams['font.size'] = 15
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

result = pickle.load(open('./results/drosophila_J_dim.p', 'rb'))
lin_dim, nonlin_dim, lins, nonlins = [result[k] for k in ['lin_dim', 'nonlin_dim', 'lins', 'nonlins']]

# %%
for dat in [(nonlins, nonlin_dim), (lins, lin_dim)]:
    for i in range(2):
        plt.figure()
        plt.hist(dat[0][:, i], bins = 10)
        plt.axvline(dat[1], color = 'k')
        plt.show()

# %% try decorrelating J
# what is the inpur covariance that would be optimally decorrelated by the experimentally observed J?

J_G = np.load('niks_crap/glom_J.npy')
C = J_G @ J_G.T

plt.figure()
plt.imshow(C)
plt.show()

vals,Z1 = np.linalg.eigh(C)

print(Z1.shape)

Ctilde_int = Z1.T @ C @ Z1
plt.figure()
plt.imshow(Ctilde_int)
plt.show()

Z = Z1 * (1 / np.sqrt(vals[None, :]))
Ctilde = Z.T @ C @ Z
plt.figure()
plt.imshow(Ctilde)
plt.show()

Jtilde = Z.T @ J_G
print(models.get_dimM(Jtilde.T))

K = Z @ Z.T
plt.figure()
plt.imshow(K)
plt.show()

Xhat = np.random.normal(0, 1, (10000, 50)) @ Z.T
Yhat = Xhat @ J_G
print(models.get_dimM(Yhat))


# %%
