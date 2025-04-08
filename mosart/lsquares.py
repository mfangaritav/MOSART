import numpy as np
import matplotlib.pyplot as plt
import h5py
import subprocess
import time
from scipy.optimize import minimize
import os
import importlib
from scipy.interpolate import interp1d
from scipy.signal import find_peaks,butter, lfilter, freqz,filtfilt
import multiprocessing
from functools import partial


def getdem(key,h5file='descending.h5',weight=None,water=None):
    """Calculates elevation map from amplitude image

    Args:
        key: numeric key with date of amplitude image
        h5file: input h5 file with amplitude, dem and longitude and latitude data
        weight: array with weights for the pixels in the inversion
        water: mask with that indicate water pixels

    Returns:
        Derived elevation map and gradient and weights
    """
    h5i = h5py.File(h5file,'r')
    dem=h5i['dem'][:]
    grd=h5i['grad'][:]
    amp=h5i[key][:]
    h5i.close()
    
    print('Start DEM for', key)
    if water is not None:
        for i in range(amp.shape[0]):
            waterf=water[i,:]
            grdf=grd[i,:]
            ampf=amp[i,:]
            cond=np.logical_and(np.logical_not(waterf),np.abs(grdf)<0.01)
            if np.sum(cond)>0:
                value=np.median(ampf[np.logical_and(np.logical_not(waterf),np.abs(grdf)<np.percentile(np.abs(grdf),5))])
                amp[i,:][waterf]=value
    demdef,adef,bdef,cdef,weights=getabc_reg(amp,dem,grd,weight)
    print('Finished DEM for', key)
    
    return demdef,amp*adef+bdef,weights


def getgrad(amps,grd,mask=None):
    """Calculates gradient of elevation from amplitude image

    Args:
        amps: input amplitude image
        grd: gradient of predefined DEM
        mask: mask that indicates what pixels to omit in the inversion

    Returns:
        Derived gradient
    """
    if mask is None:
        mask=(np.zeros(amps.shape)==1)
    grddef=np.ones(amps.shape)*np.nan
    for i in range(amps.shape[0]):
        ampf=amps[i,:]
        grdf=grd[i,:]
        maskf=mask[i,:]
        G=np.zeros((np.sum(~maskf),2))
        data=grdf[~maskf]
        G[:,0]=ampf[~maskf]
        sol = np.linalg.lstsq(G, data, rcond=None)[0]
        grddef[i,:]=sol[0]*ampf+sol[1]
    
    return grddef


def getabc_reg(amps,dem,grd,weight=None):
    """Inversion to calculate values to scale amplitude image into gradient (of elevation) values and integration constants

    Args:
        amps: input amplitude image
        dem: predefined DEM
        grd: gradient of predefined DEM
        weight: array with weights for the pixels in the inversion

    Returns:
        Derived elevation map, values to scale amplitude image, integration constants and weights
    """
    ampcp=np.copy(amps)
    demcp=np.copy(dem)
    grdcp=np.copy(grd)
    if np.sum(np.isnan(ampcp))>0:
        mask1 = np.isnan(ampcp)
        ampcp[mask1] = np.interp(np.flatnonzero(mask1), np.flatnonzero(~mask1), ampcp[~mask1])
    if np.sum(np.isnan(grdcp))>0:
        mask1 = np.isnan(grdcp)
        grdcp[mask1] = np.interp(np.flatnonzero(mask1), np.flatnonzero(~mask1), grdcp[~mask1])
    if np.sum(np.isnan(demcp))>0:
        mask1 = np.isnan(demcp)
        demcp[mask1] = np.interp(np.flatnonzero(mask1), np.flatnonzero(~mask1), demcp[~mask1])
    adef=np.ones(dem.shape)*np.nan
    bdef=np.ones(dem.shape)*np.nan
    cdef=np.ones(dem.shape)*np.nan
    weights=np.ones(dem.shape)*np.nan
    demdef=np.ones(dem.shape)*np.nan
    
    for j in range(ampcp.shape[0]):
        ampf=np.copy(ampcp[j,:])
        grdf=np.copy(grdcp[j,:])
        demf=np.copy(demcp[j,:])
        
        if weight is None:
            w=ampf*0+1
        else:
            w=weight[j,:]
        weights[j,:]=w
        
        G=np.zeros((np.sum(~np.isnan(ampf)),3))
        cum=np.cumsum(ampf)
        G[:,0]=cum
        G[:,1]=(np.array(range(len(ampf)))+1)
        G[:,2]=1
        
        Gw=np.copy(G)
        Gw[:,0]=Gw[:,0]*w
        Gw[:,1]=Gw[:,1]*w
        Gw[:,2]=Gw[:,2]*w
        
        data=demf
        
        dw=data*w
        
        sol = np.linalg.lstsq(Gw.T@Gw, Gw.T@dw, rcond=None)[0]
        
        adef[j,:]=sol[0]
        bdef[j,:]=sol[1]
        cdef[j,:]=sol[2]
        
        demdef[j,:]=np.matmul(G,sol)
        
    return demdef,adef,bdef,cdef,weights

def get_timeseries(demdefs,lam=0.001):
    """Inversion to calculate elevation changes using SBAS strategy

    Args:
        demdefs: elevation maps result of the inversion
        lam: regularization constant
        
    Returns:
        Timeseries with elevation changes
    """
    demdefs=np.array(demdefs)
    difs=[]
    indexes=[]
    for i in range(demdefs.shape[0]-1):
        ref=demdefs[i,:,:]
        for j in range(i+1,demdefs.shape[0]):
            demdef=demdefs[j,:,:]
            difs.append(demdef-ref)
            indexes.append((i,j))
    difs=np.array(difs)

    timeseries=np.ones(demdefs.shape)*np.nan
    for i in range(demdefs.shape[1]):
        for j in range(demdefs.shape[2]):
            G=np.zeros((difs.shape[0]+demdefs.shape[0],demdefs.shape[0]))
            d=np.zeros((difs.shape[0],))
            for m,index in enumerate(indexes):
                d[m]=difs[m,i,j]
                k,l=index
                G[m,k]=-1
                G[m,l]=1

            sol = np.linalg.lstsq(G.T@G, G.T@d, rcond=None)[0]
            timeseries[:,i,j]=sol
    return timeseries