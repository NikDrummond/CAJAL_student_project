#%%

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

#%%
N = 50 #MB
Ks = np.arange(2,21) #number of inputs per KC
S = 14000 #total number of connections
Ms = np.round(S/Ks, 0).astype(int) #number of KCs
d1s = 1 / (1/Ms + 1/N*(1+(Ks-1)**2/N)) #no inhibition
KN2 = Ks**2*(1-Ks/N)**2
d2s = Ms*KN2/(KN2+(Ms-1)*(Ks**2/N*(1-Ks/N)*(N-Ks)/(N-1))) #inhibition

plt.figure()
plt.plot(Ks, d1s)
plt.plot(Ks, d2s)
plt.xlabel('K')
plt.ylabel('input current dim')
plt.show()

#%%
T = 10000
f = 0.1
rep = 3

dims = np.zeros((len(Ms), 2))
for iK in range(len(Ks)):

    K, M = Ks[iK], Ms[iK]

    X = np.random.normal(0, 1, (rep, T, N))
    J = np.zeros((rep, N, M))
    for r in range(rep):
        for i in range(M):
            J[r, np.random.choice(N, K, replace = False), i] = 1

    Ye = X@J
    Yi = Ye - K*np.mean(X, axis = -1, keepdims = True)

    for iinh, Y in enumerate([Ye, Yi]):
        thresh = np.quantile(Y, 1-f, axis = 1, keepdims = True)
        R = 0.5*np.sign(Y - thresh)+0.5
        R = R - np.mean(R, axis = 1, keepdims = True)
        C = R.transpose(0,2,1) @ R
        evals = np.linalg.eigvalsh(C)
        dim = np.sum(evals, axis = -1)**2 / np.sum(evals**2, axis = -1)
        dims[iK, iinh] = np.mean(dim)

    print(K, dims[iK, :]/N)

# %%

print(Ks[np.argmax(dims, axis = 0)])
print(np.amax(dims, axis = 0)/N)

plt.figure(figsize = (4.5,4))
plt.plot(Ks, dims[:, 0]/N, '-o', color = 'k')
plt.plot(Ks, dims[:, 1]/N, '-o', color = 'r')
plt.xlabel('K')
plt.ylabel('Dimensionality/N')
plt.ylim(0, np.amax(dims)/N*1.1)
plt.title('Mushroom body', pad = 20)
plt.xticks([0,5,10,15,20])
plt.xlim(0, 21)
plt.yticks(range(7))
plt.ylim(0, 6)
plt.show()

# %%
