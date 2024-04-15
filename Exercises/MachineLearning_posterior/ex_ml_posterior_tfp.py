#%% Computing posterior statistics form the posterior distribution using machine learning

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% functions
# Define learning-rate scheduler
def scheduler(epoch,lr):
    epoch_start = 10
    if epoch < epoch_start:
        return lr
    else:
        return lr * tf.math.exp(-0.005)

# Define negative log-likelihood (used as loss function)
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

#%% Load data

# Set training data size to use
N_use = 100000

# hdf5 file with training data
file_training = '1D_P51_NO500_451_ABC5000000_0000_D2_HTX1_1.h5'
# hdf5 file with real data
file_sampling = '1D_P51_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5'

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


# %% Split data into training and validation sets
m_train, m_test, d_train, d_test = train_test_split(M,D, test_size=0.33, random_state=42)
print(m_train.shape)
print(d_train.shape)
print(m_test.shape)
print(d_test.shape)
    

# %% Setup Neural Network

# Network structure
act = 'relu'  # See https://www.tensorflow.org/api_docs/python/tf/keras/activations
nunits = 30
nhidden = 4
pdropout = 0 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(nunits, input_dim=nd, activation=act))
for i in range(nhidden-1):
    model.add(tf.keras.layers.Dense(nunits, activation=act)) 
    if (pdropout>0):
        model.add(tf.keras.layers.Dropout(pdropout))

# 1D as output normal distribution
d_floor = 0.1#1e-3 
d_scale=1
model.add(tf.keras.layers.Dense(nm+nm))
model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t[..., :nm],
                                                                                   scale=d_floor + tf.math.softplus(d_scale * t[..., nm:])))),


# Optimizations algorithm
learning_rate=1e-3
optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
# Loss function
model.compile(optimizer=optim, loss='mean_squared_error')   

# Show model summary
model.summary()



#%% Train the model
nepochs = 10
batch_size = 512
patience = 50 # max number of itarations with no improvement in validation loss
use_learningrate_schedule = False

modelname = '%s_nu%d_nh%d_pd%d_ep%d_bs%d_p%d_lr%d' % (act, nunits, nhidden, pdropout*100,nepochs,batch_size,patience,use_learningrate_schedule)
print("training %s " % (modelname))

# Callback -- Tensorboard
logdir = os.path.join("logs",modelname)
print("Logs in %s" % (logdir) )
print("Run tensorboard --logdir logs/")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)

# Early stopping
earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=3)

# Learningrate schedule
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

if use_learningrate_schedule:
    callbacks = [tensorboard_callback,earlystopping_callback,lr_callback]
else:   
    callbacks = [tensorboard_callback,earlystopping_callback]


# Train the model
out = model.fit(d_train, m_train, 
    epochs=nepochs, 
    batch_size=batch_size,
    verbose=1,
    validation_data=(d_test,m_test),
    callbacks=callbacks,
    )

# %% Plot loss
plt.figure(1)
plt.semilogy(out.history['loss'], label='Train')
plt.semilogy(out.history['val_loss'], label='Validation')
plt.xlabel('Iteration #')
plt.ylabel('Loss')
plt.grid()
plt.legend()

# %% Use the trained network for prediction
t0=time.time()
POST = model(D_obs)
M_ml_mean = POST.mean().numpy().T
M_ml_std = POST.stddev().numpy().T
t1=time.time()
t_pred = t1-t0
print('Time elapsed for prediction: %f3.3 s' % t_pred)

plt.figure(2)
plt.subplot(4,1,1)
plt.imshow(M_ml_mean, vmin=-1, vmax=3, cmap='jet')
plt.colorbar(label='log_10(ρ)')
plt.xlabel('Location #')
plt.ylabel('Depth')
plt.title('Estimated mean using ML regression')

plt.subplot(4,1,2)
plt.imshow(M_ml_std, vmin=0, vmax=1, cmap='gray_r' )
plt.colorbar(label='log_10(ρ)')
plt.xlabel('Location #')
plt.ylabel('Depth')
plt.title('Estimated std using ML regression')


plt.subplot(4,1,3)
plt.imshow(M_est_mean.T, vmin=-1, vmax=3, cmap='jet')
plt.colorbar(label='log_10(ρ)')
plt.xlabel('Location #')
plt.ylabel('Depth')
plt.title('Estimated mean using rejection sampling')

plt.subplot(4,1,4)
plt.imshow(M_est_std.T, vmin=0, vmax=1, cmap='gray_r' )
plt.colorbar(label='log_10(ρ)')
plt.xlabel('Location #')
plt.ylabel('Depth')
plt.title('Estimated std using rejection sampling')

# %%
