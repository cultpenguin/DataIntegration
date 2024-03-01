#%% Simple sampling
import matplotlib.pyplot  as plt
import numpy              as np

from f_blind import f_blind

#%% Evaluating f_blind
f_max = 1

# Propose a model from a uniform proposal distribution
x_propose = 100 * np.random.random()

# Compute the pdf of the proposed model
f_propose = f_blind(x_propose)

# Compute the acceptance probability
P_acc = f_propose/f_max

m_acc = []

# Accept (or reject) the proposed model proportional to the acceptance probability P_acc
r=np.random.random()
if r<P_acc:
    m_acc.append(x_propose)
    print('ACCEPT')
    

