#%% Simple sampling
import matplotlib.pyplot  as plt
import numpy              as np

from f_blind import f_blind

#%% Evaluating f_blind

#%%
def draw_fig():

    # Cpomute histogram
    [h,hx]   = np.histogram(m_acc,np.arange(0,51,1))
    hx = (hx[1:] + hx[:-1])/2
    hsum     = np.sum(h*(hx[1]-hx[0]))
    hh       = np.divide(h,float(hsum))
    
    f, ax = plt.subplots(1,2)
    ax[0].bar(hx,hh)
    #ax[0].set_xlim([0, 100])
    ax[0].set_ylim([0, 0.07])
    ax[0].set_title('Histogram of accepted values')
    ax[0].set_xlabel('m')
    ax[0].set_ylabel('f(m), Frequency')

    ax[1].plot(m_acc,'.')
    #ax[1].set_xlim([0, 1500])
    ax[1].set_ylim([0, 100])
    ax[1].set_title('Appectance frequency = %6.2f %%'%(100*float(i_acc)/float(i)))
    ax[1].set_xlabel('Iteration number')
    ax[1].set_ylabel('m^*')
    f.tight_layout()


#%%  Rejection sampling

f_max   = .1

m_acc   = []
i_acc   = 0
N       = 10000


for i in range(N):
   m_pro    = np.random.rand()*100
   f        = f_blind(m_pro)
   P_acc    = f/f_max
   
   # accept with probability P_acc
   if np.random.rand() < P_acc:
       i_acc = i_acc+1
       m_acc.append(m_pro)
       
   
   if np.mod(i+1,np.ceil(N/4))==0 and i > 0:
        draw_fig() 
       
draw_fig()       

# %%
