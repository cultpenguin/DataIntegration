#%% Computing posterior statistics form the posterior distribution using machine learning

import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.ion()

#%% Load data

# Set training data size to use
N_use = 100000

# hdf5 file with training data
file_training = '1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1.h5'
# hdf5 file with real data
file_sampling = '1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5'

f_training = h5py.File(file_training, 'r')
f_sampling = h5py.File(file_sampling, 'r')

N = f_training['M1'].shape[0]
if N_use > N:
    print('Requesting larger training data set than available (%d>%d)' % (N_use,N))

print("Reading training data from %s" % (file_training))    
M = f_training['M1'][0:N_use,:]
D = f_training['D2'][0:N_use,:]

print("Reading data from %s" % (file_sampling))    
D_obs= f_sampling['D_obs'][:]

# For comparison, we also load the mean and standard deviation of the posterior distribution as obtained using rejection sampling
M_est_mean= f_sampling['M_est'][:]
M_est_std= f_sampling['M_std'][:]

# Close hdf5 files
f_training.close()
f_sampling.close()

# plot model and data
print("Shape of M:", M.shape)
print("Shape of D:", D.shape)
print("Shape of D_obs:", D_obs.shape)

nd=D.shape[1]
nm=M.shape[1]
print('nm= %d' % nm)
print('nd= %d' % nd)
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(M[0:10,:].T)
plt.title('model realization')
plt.xlabel('depth')
plt.ylabel('resistivity (log10)')
plt.subplot(1,2,2)
plt.title('data (with noise) realizations')
plt.plot(D[0:10,:].T)
plt.xlabel('data #')
plt.ylabel('data value')
plt.tight_layout()
plt.show()


# %% Split data into training and validation sets
m_train, m_test, d_train, d_test = train_test_split(M,D, test_size=0.33, random_state=42)
print(m_train.shape)
print(d_train.shape)
print(m_test.shape)
print(d_test.shape)
    

# %% Setup Neural Network

# Network structure
act = 'relu'
nunits = 30
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(nunits, input_dim=nd, activation=act))
model.add(tf.keras.layers.Dense(nunits, activation=act)) 
model.add(tf.keras.layers.Dense(nunits, activation=act)) 
model.add(tf.keras.layers.Dense(nunits, activation=act)) 
model.add(tf.keras.layers.Dense(nm))

# Optimizations algorithm
learning_rate=1e-3
optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
# Loss function
model.compile(optimizer=optim, loss='mean_squared_error')   

# Show model summary
model.summary()

#%% Train the model
nepochs = 100
batch_size = 32
out = model.fit(d_train, m_train, 
    epochs=nepochs, 
    batch_size=batch_size,
    verbose=1,
    validation_data=(d_test,m_test),
    )

# %%
# Plot loss
plt.figure(2)
plt.semilogy(out.history['loss'], label='Train')
plt.semilogy(out.history['val_loss'], label='Validation')
plt.xlabel('Iteration #')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

# %%
M_EST = model.predict(D_obs)


plt.figure(3)
plt.imshow(M_EST.T, vmin=-1, vmax=3, )
plt.colorbar(label='log_10(ρ)')
plt.xlabel('Location #')
plt.ylabel('Depth')
plt.show()
# %%
