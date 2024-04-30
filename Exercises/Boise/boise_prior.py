# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:14:50 2020

@author: thoma
"""

#%%
import pygstat as pygstat
import numpy as np
import matplotlib.pyplot as plt
import time



from pygstat import boise_prior


#%%
'''
def boise_prior(m_cur=[], step=1, O={}):
    if len(O)==0:
        
        dx=0.25;
        O['x']=np.arange(0,44,1)*dx
        O['y']=np.arange(0,66,1)*dx
        O['nx']=len(O['x'])
        O['ny']=len(O['y'])
        O['nxy']=O['nx']*O['ny']
        O['v0']=0.0851;
        
        Cm='0.0000078673 Exp(10,110,0.5)'
        xx, yy = np.meshgrid(O['x'],O['y'])
        pos=np.array([xx.flatten(), yy.flatten()]).transpose()
        Cmat = pygstat.precalc_covariance(pos,pos, Cm)
        O['Ll'] = np.linalg.cholesky(Cmat)

    r=np.random.randn(O['nxy'], 1) 
    m_pro = O['Ll']@r+  O['v0']
    m_pro = m_pro.reshape(O['ny'],O['nx'])

    # random walk ??
    if (len(m_cur)>0)&(step>0):
         m_pro = O['v0']  + (m_cur-O['v0'])*np.cos(step*np.pi/2) +  (m_pro-O['v0'])*np.sin(step*np.pi/2); 

    return m_pro, O
'''
#%%   

m_pro, O = boise_prior()    

plt.subplot(1,2,1)
plt.imshow(m_pro)
plt.colorbar()

step=.01
m_pro, O = boise_prior(m_pro,step,O)    
plt.subplot(1,2,2)
plt.imshow(m_pro)
plt.colorbar()



'''    
step=0;

O={}

if len(O)==0:
    
    dx=0.25;
    O['x']=np.arange(0,44,1)*dx
    O['y']=np.arange(0,66,1)*dx
    O['nx']=len(O['x'])
    O['ny']=len(O['y'])
    O['nxy']=O['nx']*O['ny']
    O['v0']=0.0851;
    
    Cm='0.0000078673 Exp(10,110,0.5)'
    xx, yy = np.meshgrid(O['x'],O['y'])
    pos=np.array([xx.flatten(), yy.flatten()]).transpose()
    Cmat = pygstat.precalc_covariance(pos,pos, Cm)
    O['Ll'] = np.linalg.cholesky(Cmat)

r=np.random.randn(O['nxy'], 1) 
m_pro = O['Ll']@r+  O['v0']
m_pro = m_pro.reshape(O['ny'],O['nx'])


print('Elapsed time: %g s' % (t1-t0))
print('Elapsed time: %g s' % (t2-t1))


plt.imshow(m_pro)
plt.colorbar()


#precalc_covariance(pos1,pos2,V='1 Exp(10)')
'''