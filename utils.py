import numpy as np

def get_anal_rho(mu, var, N):
  vs = var + mu**2
  ss = vs**2 / (N*(N-1)) + mu**2 / (N-1) - 2 * vs * mu / (N*(N-1)) + mu**4 / N**2 - 2*mu**2*vs/N**2
  return ss

def get_Sii_2(mu, var, N):
  vs = var + mu**2
  S_ii_2 = vs*(1 + 4*mu**2 / N**2 - 4*mu/N) + 2*mu**3/N - 3*mu**4/N**2
  return S_ii_2

def get_anal_dim(mu, var, N, approx = True, M = None):
  S_ij_2 = get_anal_rho(mu, var, N)
  S_ii = mu*(1 - mu/N)
  if approx: #large M limit
    dim = S_ii**2 / S_ij_2
  else:
    S_ii_2 = get_Sii_2(mu, var, N)
    dim = M*S_ii**2 / (S_ii_2 + (M-1) * S_ij_2)
  return dim

def get_opt_var(mu, N):
  return mu*(1 - mu/N)

def get_opt_dim_var(mu, M, N):
  return get_opt_var(mu, N) - N*(N-1)/(2*M-2) * (1 + 4*(mu**2/N**2) - 4*mu/N)

