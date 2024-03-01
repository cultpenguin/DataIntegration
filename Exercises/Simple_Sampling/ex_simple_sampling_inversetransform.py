# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sampling from a probability distribution
#
# The following exercise involved sampling, i.e. generating a sample, from a probability distribution $f(x)$, both in case $f(x)$, $F(x)$, and $F^{-1}(x)$ is known, and in case $f(x)$ is unknown, but where an algorithm exists that can evaluate $f(x)$ for any value of $x$.
#

# %% [markdown]
# ## 1. Inverse transform sampling  using $F^{-1}$ / The inversion method

# %% [markdown]
# ### 1A Inverse sampling of a Gaussian distribution

# %% [markdown]
# Consider the 1D Gaussian distribution $N(8,3)$. Use Inverse Transform sampling (the 'inversion'  method) to generate a sample, $\mathbf{x}^*$, consisting of 10000 realizations of $f(x)$~$N(8,3)$.

# %%
import matplotlib.pyplot  as plt
import numpy              as np
import scipy              as sp
#import time               as time
#import random             as random

#from scipy.stats         import norm, invgauss
#from scipy.interpolate   import interp1d

# %%
mu  = 8.
var = 3.
s   = np.sqrt(var)

N   = 10000                 # How many realisations (sample size)
M   = 1                   # How many 'chains'

p   = np.random.rand(M,N)   # (M x N)-matrix, containing random numbers between 0 and 1
m   = sp.stats.norm.ppf(p,mu,s)      # 'Inversion method' applied on p


means = np.mean(m, axis=0)
stdevs = np.std(m, axis=0)

# compute mean as a function of sample size
m_mean = m*0
m_std = m*0
for j in range(M):
    for i in range(N):
        m_mean[j,i]=np.mean(m[j,0:i])
        m_std[j,i]=np.std(m[j,0:i])



# %%
plt.subplot(2,1,1)
plt.semilogx(m_mean.T)
plt.plot([0,N],[mu,mu],'k:')
plt.ylabel('Mean')
plt.grid()
plt.subplot(2,1,2)
plt.semilogx(m_std.T)
plt.plot([0,N],[s,s],'k:')
plt.grid()
plt.ylabel('Standard deviation')
plt.xlabel('Iteration Number')


# %% [markdown]
# ### Compute the probability that $x$ is larger than 13, i.e. $P(x>13)$.

# %%
# Compuer ratio of entries on m larger than 13
P_13 = ratio = np.sum(m>13)/m.size
print('P(m>13) =',P_13)


# %% [markdown]
# ## 1B 
