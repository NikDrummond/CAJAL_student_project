import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import erfc, erfcinv
from numpy.polynomial.hermite import hermgauss

def get_J(params, rng, distribution, inh_flag):
 
    M = np.floor(params['S'] / params['K']).astype(int)  # find M according to the mean K (specified in params['K'] is the mean K)
    J = np.zeros((params['N'], M))                       # initialize weight matrix

    # get J with each indegree K according to distribution
    if distribution == 'constant':
        J = __constantK__(params, rng, J, M)
    elif distribution == 'step':
        J = __stepK__(params, rng, J, M)
    elif distribution == 'binomial':
        J = __binomialK__(params, rng, J, M)
    elif distribution == 'poisson':
        J = __poissonK__(params, rng, J, M)
    elif distribution == 'lognormal':
        J = __lognormalK__(params, rng, J, M)
    elif distribution == 'gaussian':
        J = __gaussianK__(params, rng, J, M)
    else:
        raise RuntimeError("Distribution is not in the list")

    if inh_flag:
        J = J - np.mean(J)

    return J


def __constantK__(params, rng, J, M):
    for i in range(M):
        J[rng.choice(params['N'], params['K'], replace=False), i] = 1
    return J

def __stepK__(params, rng, J, M):
    for i in range(M):
        Ki = params['K'] + params['stepK']*((i%3)-1) # change K between 3 values: [K-stepK, K, K+stepK]
        J[rng.choice(params['N'], Ki, replace=False), i] = 1
    return J

def __binomialK__(params, rng, J, M):
    n = np.ceil((params['K']-1)/params['binomial_p'])  # define n for binomial distribution
    p = (params['K'] - 1)/n                            # define p for binomial, correct it according to n
    for i in range(M):
        Ki = np.floor(rng.binomial(n, p)).astype(int)
        J[rng.choice(params['N'], Ki, replace=False), i] = 1
    return J

def __poissonK__(params, rng, J, M):
    for i in range(M):
        Ki = np.floor(rng.poisson(params['K'],1)).astype(int)       # the parameter K is the mean
        while Ki < 0:                                               # to remove negative indegree values
            Ki = np.floor(rng.poisson(params['K'],1)).astype(int)
        J[rng.choice(params['N'], Ki, replace=False), i] = 1
    return J

def __gaussianK__(params, rng, J, M):
    for i in range(M):
        Ki = np.floor(rng.normal(params['K'], params['gaussian_stdK'], 1)).astype(int)       # the parameter K is the mean, gaussian_stdK is the standard dev
        while Ki < 0:                                                                        # to remove negative indegree values
            Ki = np.floor(rng.normal(params['K'], params['gaussian_stdK'], 1)).astype(int)
        J[rng.choice(params['N'], Ki, replace=False), i] = 1
    return J

def __lognormalK__(params, rng, J, M):
    for i in range(M):
        Ki = np.floor(rng.lognormal(params['K'], params['lognormal_stdK'], 1)).astype(int)       # the parameter K is the mean, lognormal_stdK is the standard dev
        while Ki < 0:                                                                            # to remove negative indegree values
            Ki = np.floor(rng.lognormal(params['K'], params['lognormal_stdK'], 1)).astype(int)
        J[rng.choice(params['N'], Ki, replace=False), i] = 1
    return J



def get_input(params, rng):
    X = rng.normal(0, 1, (params['T'], params['N']))
    return X

def run_model(params, J, X):
    H = np.matmul(X, J)
    sigs = np.sqrt(np.sum(J**2, axis=0, keepdims=True))
    thresholds = np.sqrt(2)*sigs*erfcinv(2*params['f'])
    rates = 0.5*np.sign(H - thresholds) + 0.5
    rates -= np.mean(rates, axis=0, keepdims=True)
    return rates

def get_dimM(rates):
    C = np.cov(rates)
    trC = np.trace(C)
    dim = trC**2/np.sum(C*C)
    return dim

def simulate(params, distribution):
    
    # run for multiple trials, inh vs no inh
    dims = np.zeros((params['reps'], 2))

    for rp in range(params['reps']):
        for iinh in range(2):
            rng = np.random.default_rng()
            J = get_J(params, rng, distribution, bool(iinh))
            X = get_input(params, rng) 
            rates = run_model(params, J, X)
            dims[rp, iinh] = get_dimM(rates)/params['N']

    return dims
