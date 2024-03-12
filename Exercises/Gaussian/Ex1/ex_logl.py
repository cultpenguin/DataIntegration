# %%
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

# %%
file_mat = 'logl_test_data.mat'
# list all varaiables in Matlab file file_mat
sp.io.whosmat(file_mat)

MAT = sp.io.loadmat(file_mat)
d0 = MAT['d0_1'].flatten()
dstd = MAT['dstd_1'].flatten()


def loglikelihood(d, d_obs, d_std):
    f = -0.5 * np.sum((d_obs - d) ** 2 / (d_std ** 2))
    return f



# %% Generate sample
nd = d0.shape[0]
d_sample = np.zeros((nd,1000))
logL_sample = np.zeros((1000))
N=1000
for i in range(N):
    # generate a sample from N(d0,dstd)
    d_real = np.random.normal(d0,dstd)
    d_sample[:,i] = d_real
    logL_sample[i]=loglikelihood(d_real, d0, dstd)    


# %%

# %%
