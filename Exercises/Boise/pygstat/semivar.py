# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 09:40:38 2018

@author: thoma
"""
import numpy as np
import matplotlib.pyplot as plt;

# gslib of gstat format?
gstat=0



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
        Cmat = precalc_covariance(pos,pos, Cm)
        O['Cm']=Cmat
        O['Ll'] = np.linalg.cholesky(Cmat)

    r=np.random.randn(O['nxy'], 1) 
    m_pro = O['Ll']@r+  O['v0']
    m_pro = m_pro.reshape(O['ny'],O['nx'])

    # random walk ??
    if (len(m_cur)>0)&(step>0):
         m_pro = O['v0']  + (m_cur.copy()-O['v0'])*np.cos(step*np.pi/2) +  (m_pro.copy()-O['v0'])*np.sin(step*np.pi/2); 

    return m_pro, O

   



def precalc_covariance(pos1,pos2,V='1 Exp(10)'):
    '''Precalculate covariance model
    '''
    n1=pos1.shape[0]
    n2=pos2.shape[0]

    # MUCH FASTER
    D=np.zeros([n1,n2])
    for i in range(n1):
        D[i] = np.linalg.norm(pos1[i]-pos2,axis=1)
    G = global_variance(V) - semivariance(V,D)
    
    """
    # MUCH SLOWER
    G=np.zeros([n1,n2])
    for i in range(n1):
        for j in range(n2):
            #D[i,j] = np.linalg.norm(pos1[i]-pos2[j])
            #G[i,j] = global_variance(V) - semivariance(V,D[i,j])
            D = np.linalg.norm(pos1[i]-pos2[j])
            G[i,j] = global_variance(V) - semivariance(V,D)
    """ 
    return G
    
def global_variance(V):
    '''global_variance 
    Get global variance from variogram model
    '''
    V = txt_to_variogram(V)
    gvar = 0
    for i in range(len(V)):
        gvar = gvar + V[i]['sill']
    return gvar
    
def semivariance(V=[{}],hx=0,gstat=gstat):
    
    # convert to variogram structure if string
    if isinstance(V, str):
        V = txt_to_variogram(V)    
    
    nV = len(V)
    
    gvar = 0;
    for i in range(nV):
        gvar = gvar + V[i]['sill']
    
    s = 0
    for i in range(nV):
        sgamma = hx*0.0
        #print("V = %s" % variogram_to_txt([V[i]]) )
        if V[i]['type'].lower()=='nug':
            sgamma = V[i]['sill']+hx*0
        
        elif V[i]['type'].lower()=='sph':     
            '''
            Spherical type semivariance
            '''
            sgamma=V[i]['sill']*(1.5*np.abs(hx)/V[i]['range'][0]-0.5*(hx/V[i]['range'][0])**3)
            #ii=np.argwhere(hx>V[i]['range'][0])
            if (not isinstance(hx,np.ndarray))&(len(np.atleast_1d(hx))==1):
               if (hx>V[i]['range']):
                   sgamma=V[i]['sill']        
            else:
                ii=hx>V[i]['range']
                sgamma[ii]=V[i]['sill']
            
        elif V[i]['type'].lower()=='exp':
            '''
            Exponential type semivariance
            '''
            sgamma = 0
            if gstat==0:
                sgamma=V[i]['sill']*(1-np.exp(-3*hx/V[i]['range'][0])) # GSLIB2/Goovaerts
            else:
                sgamma=V[i]['sill']*(1-np.exp( -hx/V[i]['range'][0])) # GSTAT
            
        elif V[i]['type'].lower()=='gau':     
            '''
            Gaussian type semivariance
            '''
            sgamma = 0
            if gstat==0:
                sgamma=V[i]['sill']*(1-np.exp(-3*hx**2/V[i]['range'][0]**2)) # GSLIB2/Goovaerts
            else:
                sgamma=V[i]['sill']*(1-np.exp( -hx**2/V[i]['range'][0]**2)) # GSTAT
            
        
        #ii=np.argwhere(hx<1e-19)
        if (not isinstance(hx,np.ndarray))&(len(np.atleast_1d(hx))==1):
            if (hx<1e-19):
                sgamma=0
        else:
            ii=np.nonzero(hx<.1e-19)
            sgamma[ii]= 0
            
        
        s=s+sgamma
    
    return s


def variogram_to_txt(V):
    '''format variogram list to txt
    '''
    
    if isinstance(V, str):
        # V is allready a string
        return V
    
    
    V_txt = '';
    for i in range(len(V)):
        
        if len(V[i]['range'])==1:
            txt  = "%g %s(%g)"  % (V[i]['sill'],V[i]['type'],V[i]['range'][0])
        elif len(V[i]['range'])==3:
            txt  = "%g %s(%g,%g,%g)"  % (V[i]['sill'],V[i]['type'],V[i]['range'][0],V[i]['range'][1],V[i]['range'][2])
        elif len(V[i]['range'])==5:
            txt  = "%g %s(%g,%g,%g)"  % (V[i]['sill'],V[i]['type'],V[i]['range'][0],V[i]['range'][1],V[i]['range'][2],V[i]['range'][3],V[i]['range'][4])
                
        V_txt = V_txt + txt
        if i<(len(V)-1):
            V_txt = V_txt + ' + '
    return V_txt



def txt_to_variogram(V_txt):
    '''Convert string to variogram list
    '''
    
    if isinstance(V_txt, list):
        # V_txt is allready a list
        return V_txt
    
    V_ind = V_txt.split(' + ')
    nV = len(V_ind)
    
    V=[]
    for i in range(nV):
        sill, rest = V_ind[i].split(' ')
        Vtype, range_tmp = rest.split('(')
        
        range_tmp2,dum = range_tmp.split(')')
        par=range_tmp2.split(',')
        
        r = np.zeros(len(par))
        for j in range(len(par)):
            r[j]=float(par[j])
    
        vv={'sill':float(sill), 'type':Vtype, 'range':r}
        V.append(vv)

    return V

