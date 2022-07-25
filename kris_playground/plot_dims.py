#%% load some libraries and set a few global parameters

import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams['font.size'] = 15
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

N = 50 # number of input glomeruli
Ks = np.arange(2,21) #number of inputs per KC to consider
S = 14000 #total number of connections (K*M)
Ms = np.round(S/Ks, 0).astype(int) #number of KCs (S/K)
f = 0.1 #coding level

#%% compute dimensionality of input current (eq 25, 28)

d1s = 1 / (1/Ms + 1/N*(1+(Ks-1)**2/N)) #no inhibition
KN2 = Ks**2*(1-Ks/N)**2
d2s = Ms*KN2/(KN2+(Ms-1)*(Ks**2/N*(1-Ks/N)*(N-Ks)/(N-1))) #inhibition

#plot result
plt.figure()
plt.plot(Ks, d1s)
plt.plot(Ks, d2s)
plt.xlabel('K')
plt.ylabel('input current dim')
plt.show()

#%% compute dimensionality of mixed representation
T = 10000 #input datapoints to consider
rep = 3 #number of repetitions (samples from the weight matrix distribution)

dims = np.zeros((len(Ms), 2)) #array for storing results
tic = time.time() #count time
for iK in range(len(Ks)): #for each K

    K, M = Ks[iK], Ms[iK]

    X = np.random.normal(0, 1, (rep, T, N)) #Gaussian inputs
    J = np.zeros((rep, N, M)) #weight matrix
    for r in range(rep): #for each repetition
        for i in range(M): #for each KC
            #randomly sample an input weight vector (this could probs be parallelized)
            J[r, np.random.choice(N, K, replace = False), i] = 1

    Ye = X@J #compute input to KCs
    #compute input with global feedforward inhibition
    Yi = Ye - K*np.mean(X, axis = -1, keepdims = True)

    #for the cases of no inhibition and inhibition
    for iinh, Y in enumerate([Ye, Yi]):
        #compute threshold dynamically according to coding level
        thresh = np.quantile(Y, 1-f, axis = 1, keepdims = True)
        #compute output of nonlinearity
        R = 0.5*np.sign(Y - thresh)+0.5
        #mean subtract to compute covariance
        R = R - np.mean(R, axis = 1, keepdims = True)
        #compute covariance matrix
        C = R.transpose(0,2,1) @ R
        #compute eigenvalues
        evals = np.linalg.eigvalsh(C)
        #compute dimensionality (participation ratio)
        dim = np.sum(evals, axis = -1)**2 / np.sum(evals**2, axis = -1)
        #store result
        dims[iK, iinh] = np.mean(dim)

    #print progress
    print(K, dims[iK, :]/N, np.round(time.time() - tic, 3))

# %% plot result

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
