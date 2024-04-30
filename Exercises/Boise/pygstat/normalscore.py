# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:35:48 2019

@author: thoma


Normal score forward and inverse transformation
"""

#%%
import numpy as np
import scipy as sp
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#%%

def inscore(d_nscore,O={}, **kwargs):
    kind = kwargs.pop('kind','linear')
    #import numpy as np
    #from scipy.stats import norm
    #from scipy.interpolate import interp1d
    if len(O)==0:
        print("Please call 'nscore' calling inscore")
        return -1
    #d_out=interp1(o_nscore.normscore(id),s_origdata(id),d_normal,style);
    f = interp1d(O['normscore'],O['sd'],  kind=kind)
    d = O['inverse'](d_nscore)
    return d




def nscore(d,O={}, **kwargs):
    import numpy as np
    from scipy.stats import norm
    from scipy.interpolate import interp1d
    
    # get infro from kwargs
    
    discrete = kwargs.pop('discrete',0)
    if discrete==1:
        kind = kwargs.pop('kind','nearest')
    else:
        kind = kwargs.pop('kind','linear')
        
    doPlot = kwargs.pop('doPlot',0)
    d_min = kwargs.pop('d_min',np.min(d))
    d_max = kwargs.pop('d_max',np.max(d))
    
    
    if len(O)==0:
        d=d.flatten()
                  
        n=len(d)
        id=np.arange(n)+1
        pk=id/n-.5/n
        normscore = norm.ppf(pk)
        sd=np.sort(d)
        
        sd_org=sd
        normscore_org=normscore
        
        if d_min<np.min(d):
            pk=np.insert(pk,0,0)
            sd=np.insert(sd,0,d_min)
        else:
            pk[0]=0;
        if d_max>np.max(d):
            pk=np.append(pk,1)
            sd=np.append(sd,d_max)
        else:
            pk[-1]=1
        
        O={'pk':pk}
        O['normscore'] = norm.ppf(pk)
        O['sd'] = sd
     
        if np.isinf(O['normscore'][0]):
            O['normscore'][0]=-10
        if np.isinf(O['normscore'][-1]):
            O['normscore'][-1]=10
            
        
    # the forwrad norlam transform
    #f = interp1d(sd_org,normscore_org)
    O['forward'] = interp1d(O['sd'],O['normscore'], kind=kind)
    
    # the inverse normal score 
    O['inverse']= interp1d(O['normscore'],O['sd'],  kind=kind, fill_value=(d_min,d_max))
    
    #f = interp1d(O['sd'],O['normscore'], fill_value='extrapolate*, kind='nearest')
    d_nscore = O['forward'](d)
    
    if doPlot==1:
        plt.figure(1)
        plt.clf()
        ax1=plt.subplot(2, 2, 1)
        ax1.hist(d,31)
        ax1.set_title('Original data')
        ax2=plt.subplot(2, 2, 2)
        ax2.hist(d_nscore,31)
        ax2.set_xlim([-4, 4])
        ax2.set_title('Normal score data')
        ax3=plt.subplot(2, 1, 2)
        ax3.plot(O['sd'],O['pk'], '*')
        ax3.set_xlabel('Original data')
        ax3.set_ylabel('Normal score data')
        ax3.set_ylim([0, 1])
        plt.show()
        
    return d_nscore, O

def plot_transform(O):
    plt.figure(2)
    plt.clf()
    ax1=plt.subplot(2, 1, 1)
    ax1.plot(O['sd'],O['pk'], '*')
    ax1.set_xlabel('Original data')
    ax1.set_ylabel('CPDF')
    ax1.set_ylim([0, 1])
    
    ax2=plt.subplot(2, 2, 3)
    ax2.plot(O['sd'],O['normscore'], '-*')
    ax2.set_xlabel('Original data')
    ax2.set_ylabel('Normal score data')
    ax2.set_ylim([-4, 4])
    
    
    ax3=plt.subplot(2, 2, 4)
    ax3.plot(O['normscore'],O['sd'], '-*')
    ax3.set_ylabel('Original data')
    ax3.set_xlabel('Normal score data')
    ax3.set_xlim([-4, 4])
    plt.show()
  