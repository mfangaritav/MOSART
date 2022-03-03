# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:01:52 2021

@author: elfer
"""
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

def onedem(i,keys,maskoutan,maskoutde,maskin,demd1,inh5):
    key=keys[i]
    demd2, grads2, ampcp2, maskoutant, maskoutdet, maskint, positivos, pshade = getdem_spr(key,maskoutan,maskoutde,maskin,demd1,h5file=inh5)

    return demd2,grads2,positivos,pshade

def getdems_sr(inh5='descending.h5',oh5='demdefs.h5'):
    h5i = h5py.File(inh5,'r')
    llaves=[key for key in h5i.keys()]
    h5i.close()
    
    demd1,grads1,ampcp1,maskoutan,maskoutde,maskin,positivos,pshade=getdem_spr(llaves[0])

    pool = multiprocessing.Pool(processes=6)
    subrutina=partial(onedem,keys=llaves,maskoutan=maskoutan,maskoutde=maskoutde,maskin=maskin,demd1=demd1,inh5=inh5)
    demds,grads,positivoss,pshades=zip(*pool.map(subrutina, range(1,len(llaves))))
    pool.close()
    pool.join()

    h5o=h5py.File(oh5,'w')
    for i in range(len(llaves)):
        key=llaves[i]
        if not i==0:
            h5o=h5py.File(oh5,'a')
        grp=h5o.create_group(key)
        if i==0:
            grp.create_dataset('demdef',data=demd1)
            grp.create_dataset('grad',data=grads1)
        else:
            grp.create_dataset('demdef',data=demds[i-1])
            grp.create_dataset('grad',data=grads[i-1])
            positivos=positivoss[i-1]
            pshade=pshades[i-1]
        grp.create_dataset('positivos',data=positivos)
        grp.create_dataset('pshade',data=pshade)
        h5o.close()
        
    amps=[]
    h5i = h5py.File(inh5,'r')
    llaves=[key for key in h5i.keys()]
    h5i.close()

    demdefs=[]
    demdefsor=[]
    changes=[]
    h5i = h5py.File(oh5,'r')
    for key in sorted(llaves):
        demdef=h5i[key+'/demdef'][:]
        demdefsor.append(np.copy(demdef))
        positivos=h5i[key+'/positivos'][:]
        pshade=h5i[key+'/pshade'][:]
        demdef1=np.ones(demdef.shape)*np.nan
        for i in range(demdef.shape[0]):
            fdem=interp1d(np.array(range(demdef.shape[1]))[~np.isnan(demdef[i,:])], demdef[i,:][~np.isnan(demdef[i,:])],kind='cubic')
            ultimo=np.array(range(demdef.shape[1]))[~np.isnan(demdef[i,:])][-1]
            demdef1[i,0:ultimo+1]=fdem(np.array(range(ultimo+1)))
        demdef[~np.isnan(positivos)]=demdef1[~np.isnan(positivos)]
        demdef=estimate_shade(demdef,pshade,maskin)
        demdefs.append(demdef)
        if key==llaves[0]:
            ref=demdef
        changes.append(demdef-ref)
    h5i.close()
    changes=np.array(changes)
    demdefs=np.array(demdefs)
    
    return demdefs,changes,maskoutan,maskoutde,maskin

def correct_dems(demdefs,h5file='descending.h5'):
    stddem=np.std(demdefs,axis=0)
    
    h5i = h5py.File(h5file,'r')
    keys=[key for key in h5i.keys()]
    key=keys[0]
    dem=h5i[key+'/dem'][:]
    h5i.close()
    
    demdefs_dr=np.copy(demdefs)
    medmaxs=getmedmaxs(demdefs)
    for k,demdef in enumerate(demdefs_dr):
        print('Correcting DEM for',keys[k])
        for i in range(demdef.shape[0]):
            ramp=getramp(demdef[i,:],dem[i,:],int(medmaxs[i]),np.argwhere(stddem[i,:]<5)[0][0])
            demdef[i,:]+=ramp
    return demdefs_dr

def getmedmaxs(demdefs):
    medmaxs=[]
    for j in range(demdefs.shape[1]):
        maximos=[]
        for demdef in demdefs:        
            maximos.append(np.nanargmax(demdef[j,:]))
        medmaxs.append(np.median(maximos))
    return medmaxs

def getramp(rowdemdef,rowdem,ini,fin):
    res=np.empty(rowdemdef.shape)
    if ini>=fin:
        res[0:fin]=np.nan
        res[fin::]=0
        return res
    else:
        slope=(rowdemdef[ini]-rowdem[ini])/(fin-ini)
        cut=-slope*fin
        res[0:ini]=np.nan
        for i in range(ini,fin):
            res[i]=slope*i+cut
        res[fin::]=0
    return res

def getabc_matrices(key,h5file='descending.h5'):

    h5i = h5py.File(h5file,'r')
    dem=h5i[key+'/dem'][:]
    grd=h5i[key+'/grad'][:]
    amp=h5i[key+'/amps'][:]
    lon=h5i[key+'/lon'][:]
    lat=h5i[key+'/lat'][:]
    std=np.load('stdamps.npy')
    h5i.close()

    shade=get_shadow(amp,lon,lat)

    ampcp=np.copy(amp)
    grdcp=np.copy(grd)

    ampcp[grdcp>10]=np.nan
    grdcp[grdcp>10]=np.nan

    fampcp=filtrar1(ampcp)
    fgrdcp=filtrar1(grdcp)
    
    a=np.empty(grd.shape)
    b=np.empty(grd.shape)
    tracesa,tracesb=getab(key,fampcp,fgrdcp,std)
    ma=np.nanmean(tracesa,axis=1)
    mb=np.nanmean(tracesb,axis=1)
    
    for i in range(grd.shape[0]):
        a[i,:]=ma[i]
        b[i,:]=mb[i]
    
    grads=(amp.T*ma).T+np.outer(mb,np.ones(fampcp[0,:].shape))
    demdef,filas,c=getdemdef1(key,dem,grads,std,lim=False)
    
    return a,b,c

def getdem(key,h5file='descending.h5'):

    h5i = h5py.File(h5file,'r')
    dem=h5i[key+'/dem'][:]
    grd=h5i[key+'/grad'][:]
    amp=h5i[key+'/amps'][:]
    lon=h5i[key+'/lon'][:]
    lat=h5i[key+'/lat'][:]
    std=np.load('stdamps.npy')
    h5i.close()

    shade=get_shadow(amp,lon,lat)

    ampcp=np.copy(amp)
    grdcp=np.copy(grd)

    ampcp[grdcp>10]=np.nan
    grdcp[grdcp>10]=np.nan

    fampcp=filtrar1(ampcp)
    fgrdcp=filtrar1(grdcp)
    
    print('Start gradient for', key)
    tracesa,tracesb=getab(key,fampcp,fgrdcp,std)
    print('Finished gradient for', key)
    ma=np.nanmean(tracesa,axis=1)
    mb=np.nanmean(tracesb,axis=1)
    
    print('Start integration for', key)
    grads=(amp.T*ma).T+np.outer(mb,np.ones(fampcp[0,:].shape))
    demdef,filas,c=getdemdef1(key,dem,grads,std,lim=False)
    print('Finished integration for', key)
    
    return demdef

def calc_std(h5file='descending.h5'):
    h5i=h5py.File(h5file,'r')
    llaves=sorted([key for key in h5i.keys()])
    h5i.close()
    amps=[]
    for i,key in enumerate(llaves):
        h5i = h5py.File(h5file,'r')
        ampscp=h5i[key+'/amps'][:]
        h5i.close()
        amps.append(ampscp)

    amps=np.array(amps)
    std=np.nanstd(amps,axis=0)
    del amps
    del ampscp
    np.save('stdamps.npy',std)

def get_masks(grads):
    maskin=np.ones(grads.shape)
    maskoutan=np.ones(grads.shape)
    maskoutde=np.ones(grads.shape)
    signos=grads/np.abs(grads)
    outsignos=np.ones(grads.shape)
    outsignos[:,0:signos.shape[1]-1]=signos[:,0:signos.shape[1]-1]*signos[:,1:signos.shape[1]]

    ini=int(outsignos.shape[0]/2)
    for i in range(int(outsignos.shape[0]/2)+1):
        j=ini-i
        i=i+ini
        posnegar=np.argwhere(outsignos[i,:]<0)
        posnegar=np.array([pos[0] for pos in posnegar])
        posnegab=np.argwhere(outsignos[j,:]<0)
        posnegab=np.array([pos[0] for pos in posnegab])
        if i==ini:
            pos0ar=posnegar[0]
            pos1ar=posnegar[-1]
            pos0ab=posnegab[0]
            pos1ab=posnegab[-1]
        else:
            if len(posnegab)>0:
                min0=np.min(np.abs(posnegab-pos0ab))
                min1=np.min(np.abs(posnegab-pos1ab))
                if min0<100 and min1<100: 
                    poss0=np.abs(posnegab-pos0ab)
                    poss1=np.abs(posnegab-pos1ab)
                    if not np.argmin(poss0)==np.argmin(poss1):
                        pos0ab=posnegab[np.argmin(poss0)]
                        pos1ab=posnegab[np.argmin(poss1)]
            if len(posnegar)>0:
                min0=np.min(np.abs(posnegar-pos0ar))
                min1=np.min(np.abs(posnegar-pos1ar))
                if min0<100 and min1<100: 
                    poss0=np.abs(posnegar-pos0ar)
                    poss1=np.abs(posnegar-pos1ar)
                    if not np.argmin(poss0)==np.argmin(poss1):
                        pos0ar=posnegar[np.argmin(poss0)]
                        pos1ar=posnegar[np.argmin(poss1)]
        maskoutan[i,pos0ar::]=np.nan
        maskoutde[i,0:pos1ar+1]=np.nan
        maskoutan[j,pos0ab::]=np.nan
        maskoutde[j,0:pos1ab+1]=np.nan
        maskin[i,0:pos0ar+1]=np.nan
        maskin[i,pos1ar::]=np.nan
        maskin[j,0:pos0ab+1]=np.nan
        maskin[j,pos1ab::]=np.nan
    return maskin,maskoutan,maskoutde

def estimate_shade(demdef,pshade,maskin):
    demdef1=np.copy(demdef)
    demdef2=np.copy(demdef)
    for i in range(demdef1.shape[0]):
        inicios,fines=get_traces(demdef1[i,:])
        for j in range(1,len(inicios)):
            demdef1[i,fines[j-1]:inicios[j]]=demdef1[i,fines[j-1]-1]+pshade[i]*np.arange(inicios[j]-fines[j-1])
    demdef2[maskin]=demdef1[maskin]
    return demdef2

def get_zeros(grads):
    zeros=np.zeros((grads.shape[0],2))
    signos=grads/np.abs(grads)
    outsignos=np.ones(grads.shape)
    outsignos[:,0:signos.shape[1]-1]=signos[:,0:signos.shape[1]-1]*signos[:,1:signos.shape[1]]
    for i in range(outsignos.shape[0]):
        posneg=np.argwhere(outsignos[i,:]<0)
        zeros[i,0]=posneg[0]
        zeros[i,1]=posneg[-1]
    return zeros

def filtrar(amp):
    famp=np.ones(amp.shape)*np.nan
    for i in range(amp.shape[0]):
        inicios,fines=get_traces(amp[i,:])
        for j in range(len(inicios)):
            data=amp[i,inicios[j]:fines[j]]
            b, a = butter(1, 0.01, 'low')
            if len(data)>6:
                famp[i,inicios[j]:fines[j]] = filtfilt(b, a, data)
            else:
                famp[i,inicios[j]:fines[j]] = filtfilt(b, a, data,padlen=0)
    return famp

def filtrar1(amp):
    famp=np.ones(amp.shape)*np.nan
    ampcp=np.copy(amp)
    for i in range(ampcp.shape[0]):
        row=ampcp[i,:][~np.isnan(ampcp[i,:])]
        b, a = butter(1, 0.01, 'low')
        if len(row)>6:
            famp[i,:][~np.isnan(ampcp[i,:])] = filtfilt(b, a, row)
        else:
            famp[i,:][~np.isnan(ampcp[i,:])] = filtfilt(b, a, row,padlen=0)
    return famp

def get_traces(row):
    posnonan=np.argwhere(~np.isnan(row))
    inicios=[]
    fines=[]
    for i,pos in enumerate(posnonan):
        if i==0:
            inicios.append(pos[0])
        elif not pos==posnonan[i-1]+1:
            inicios.append(pos[0])
            fines.append(posnonan[i-1][0]+1)
        if i==len(posnonan)-1:
            fines.append(pos[0]+1)
    '''
    for i,pos in enumerate(posnonan):
        if i==0:
            inicios.append(pos[0])
            if len(posnonan)==1:
                fines.append(pos[0]+1)
        elif i==len(posnonan)-1:
            if not pos==posnonan[i-1]+1:
                inicios.append(pos[0])
                fines.append(posnonan[i-1][0]+1)
            fines.append(pos[0]+1)
        elif not pos==posnonan[i-1]+1:
            inicios.append(pos[0])
            fines.append(posnonan[i-1][0]+1)
    '''
    #print('Trazos',len(inicios),len(fines))
    return inicios,fines

def integrate(grad):
    integration=np.ones(grad.shape)*np.nan
    for i in range(grad.shape[0]):
        inicios,fines=get_traces(grad[i,:])
        for j in range(len(inicios)):
            integration[i,inicios[j]:fines[j]]=np.cumsum(grad[i,inicios[j]:fines[j]])
    return integration

def get_shadow(amp,lon,lat):
    crater=get_ellipse(lon,lat)
    y,x=np.histogram(amp[np.isnan(crater)],bins=int(np.nanmax(amp[np.isnan(crater)])))
    peaks,_=find_peaks(y,height=500)
    #peaks,_=find_peaks(y)
    #peaks1,_=find_peaks(-y[peaks[1]::])
    peaks1,_=find_peaks(-y[peaks[0]::])
    shade=x[peaks[0]+peaks1[0]]
    #shade=x[peaks[1]+peaks1[2]]
    return shade

def get_ellipse(x,y,xcen=None,ycen=None,ax=None,ay=None):
    if not xcen:
        xcen=-163.970
    if not ycen:
        ycen=54.756
    if not ax:
        ax=0.002
    if not ay:
        ay=0.001
    elipse=np.ones(x.shape)
    elipse[np.power(x-xcen,2)/np.power(ax,2)+np.power(y-ycen,2)/np.power(ay,2)<=1]=np.nan
    #elipse[np.logical_and(x<xcen,np.power(y-ycen,2)<np.power(ay,2))]=np.nan
    return elipse

def get_mask(x):
    xcen=-163.975
    cond2=x<=xcen
    elipse=np.ones(x.shape)
    elipse[cond2]=np.nan
    return elipse

def residualgrd(x0,rowamp,rowgrd,rowstd):
    a,b=x0
    rowgrds=a*rowamp+b
    res=np.nansum(np.power(rowgrds-rowgrd,2)/rowstd)
    return res

def residualgrd_reg(x0,rowamp,rowgrd,rowstd,lamb=0):
    a,b=x0
    rowgrds=a*rowamp+b
    res=np.nansum(np.power(rowgrds-rowgrd,2)+lamb*(np.power(a,2)+np.power(b,2)))
    return res

def residualdem(b,rowsyn,rowdem,rowstd):
    rowdems=rowsyn+b[0]
    res=np.nansum(np.power(rowdems-rowdem,2)/rowstd)
    return res

def bestoffs(key,rowsyn,rowdem,rowstd):
    result = minimize(residualdem,x0=np.array([0.0]),args=(rowsyn,rowdem,rowstd), method='Nelder-Mead', tol=1e-6)
    b=result.x
    return b[0]

def bestab(key,rowamps,rowgrd,rowstd):
    result = minimize(residualgrd,x0=np.array([1.0,0.0]),args=(rowamps,rowgrd,rowstd), method='Nelder-Mead', tol=1e-6)
    a,d=result.x
    if a<0:
        a=np.abs(a)
    if d>0:
        d=d*-1
    return a,d

def bestab_reg1(key,rowamps,rowgrd,rowstd,lamb=0):
    result = minimize(residualgrd_reg,x0=np.array([1.0,0.0]),args=(rowamps,rowgrd,rowstd,lamb), method='Nelder-Mead', tol=1e-6)
    a,d=result.x
    #if a<0:
    #    print('Cambio signo a')
    #    a=np.abs(a)
    #if d>0:
    #    print('Cambio signo d')
    #    d=d*-1
    return a,d

def bestab_reg(key,rowamps,rowgrd,rowstd,lamb=0):
    pos=np.logical_and(~np.isnan(rowamps),~np.isnan(rowgrd))
    pos=np.logical_and(pos,~np.isnan(rowstd))
    A=np.ones((len(rowamps[pos]),2))
    A[:,0]=rowamps[pos]
    A[:,1]=np.ones(rowamps[pos].shape)
    sol=np.linalg.lstsq(A.T.dot(A) + lamb * np.identity(A.shape[1]), A.T.dot(rowgrd[pos]))
    a,b=sol[0]
    return a,b

def bestab1(key,rowamps,rowgrd,rowstd):
    result = minimize(residualgrd,x0=np.array([1.0,0.0]),args=(rowamps,rowgrd,rowstd), method='Nelder-Mead', tol=1e-6)
    a,d=result.x
    return a,d
'''
def bestab(key,rowamps,rowgrd,rowstd,rowmask):
    samplesa,samplesb=best_ab1.bestab(rowamps,rowgrd,rowstd,rowmask)
    importlib.reload(best_ab1)
    return samplesa,samplesb
'''

def bestabc(key,rowamps,rowdem,rowstd,rowmask):
    np.save('rowamps'+key+'.npy',rowamps)
    np.save('rowdem'+key+'.npy',rowdem)
    np.save('rowstd'+key+'.npy',rowstd)
    np.save('rowmask'+key+'.npy',rowmask)
    subprocess.call('python best_abc.py '+key,shell=True)
    samplesa=np.load('rowa'+key+'.npy')
    samplesb=np.load('rowb'+key+'.npy')
    samplesc=np.load('rowc'+key+'.npy')
    return samplesa,samplesb,samplesc

def bestramp(key,rowsyn,rowdem,rowstd,rowmask):
    np.save('rowsyn'+key+'.npy',rowsyn)
    np.save('rowdem'+key+'.npy',rowdem)
    np.save('rowstd'+key+'.npy',rowstd)
    np.save('rowmask'+key+'.npy',rowmask)
    subprocess.call('python best_ramp.py '+key,shell=True)
    samplesc=np.load('rowc'+key+'.npy')
    samplesd=np.load('rowd'+key+'.npy')
    return samplesc,samplesd

def getab(key,amps,grd,std):
    #print('Inversion ab para',key,'con',amps.shape[0],'filas')
    tracesa=np.empty((amps.shape[0],1))
    tracesa[:]=np.nan
    tracesb=np.empty((amps.shape[0],1))
    tracesb[:]=np.nan
    for i in range(amps.shape[0]):
        #if i%100==0:
        #    print('Fila AB',i)
        a,b=bestab(key,amps[i,:],grd[i,:],std[i,:])
        tracesa[i,0]=a
        tracesb[i,0]=b
    return tracesa,tracesb

def getab_reg(key,amps,grd,std,lamb=0):
    grd=np.copy(grd)
    grd[np.abs(grd)>1]=np.nan
    
    pos=np.logical_and(~np.isnan(amps),~np.isnan(grd))
    G=np.zeros((np.sum(pos)+2*amps.shape[0]-4,2*amps.shape[0]))
    data=np.zeros((np.sum(pos)+2*amps.shape[0]-4,1))
    for i in range(amps.shape[0]):
        
        if i==0:
            pos0=0
            pos1=len(amps[i,:][pos[i,:]])
        else:
            pos0=pos1
            pos1+=len(amps[i,:][pos[i,:]])
        data[pos0:pos1,0]=grd[i,:][pos[i,:]]
        G[pos0:pos1,i]=amps[i,:][pos[i,:]]
        G[pos0:pos1,amps.shape[0]+i]=np.ones(amps[i,:][pos[i,:]].shape)
        if i<amps.shape[0]-2:
            G[np.sum(pos)+i,i]=1*lamb
            G[np.sum(pos)+i,i+1]=-2*lamb
            G[np.sum(pos)+i,i+2]=1*lamb
            G[np.sum(pos)+amps.shape[0]-2+i,amps.shape[0]+i]=1*lamb
            G[np.sum(pos)+amps.shape[0]-2+i,amps.shape[0]+i+1]=-2*lamb
            G[np.sum(pos)+amps.shape[0]-2+i,amps.shape[0]+i+2]=1*lamb
    print(G.shape,data.shape)
    sol = np.linalg.lstsq(G, data, rcond=None)[0]
    trsa=np.array([sol[i] for i in range(amps.shape[0])])
    trsb=np.array([sol[i+amps.shape[0]] for i in range(amps.shape[0])])
    
    tracesa=trsa   
    tracesb=trsb
    
    return tracesa,tracesb

def getabc_reg(amps,dem):
    numpar=0
    indices=[]
    for j in range(dem.shape[0]):
        rowdem=dem[j,:]
        inicios,fines=get_traces(rowdem)
        indices.append((inicios,fines))
        numpar+=len(inicios)
    numpar+=2*dem.shape[0]
    demcp=np.copy(dem)
    data=np.zeros((np.sum(~np.isnan(dem)),1))
    G=np.zeros((np.sum(~np.isnan(dem)),numpar))
    filas=[]
    par=0
    for j in range(dem.shape[0]):
        if j%100==0:
            print('Filas offs',j)
        rowamps=amps[j,:]
        rowdem=demcp[j,:]
        inicios,fines=indices[j]
        if len(inicios)<2:
            tam=len(inicios)
        else:
            tam=2
        for i in range(tam):
            psyn=np.cumsum(rowamps[inicios[i]:fines[i]])
            pdem=rowdem[inicios[i]:fines[i]]
            pdata=pdem
            if i==0 and j==0:
                pos0=0
                pos1=len(rowdem[inicios[i]:fines[i]])
            else:
                pos0=pos1
                pos1+=len(rowdem[inicios[i]:fines[i]])
            data[pos0:pos1,0]=pdata
            G[pos0:pos1,j]=psyn
            G[pos0:pos1,dem.shape[0]+j]=np.array(range(len(psyn)))+1
            G[pos0:pos1,2*dem.shape[0]+par]=1
            par+=1
    print('Least squares')
    #print(G,data)
    #print(np.sum(np.isnan(G)),np.sum(np.isnan(data)))
    sol = np.linalg.lstsq(G, data, rcond=None)[0]
    print('Least squares finished')
    print(np.sum(np.isnan(G)))
    syndata=np.matmul(G,sol)
    print(syndata)
    par=0
    adef=np.ones(dem.shape)*np.nan
    bdef=np.ones(dem.shape)*np.nan
    cdef=np.ones(dem.shape)*np.nan
    demdef=np.ones(dem.shape)*np.nan
    for j in range(dem.shape[0]):
        inicios,fines=indices[j]
        rowdem=demcp[j,:]
        adef[j,:]=sol[j]
        bdef[j,:]=sol[dem.shape[0]+j]
        if len(inicios)<2:
            tam=len(inicios)
        else:
            tam=2
        for i in range(tam):
            if i==0 and j==0:
                pos0=0
                pos1=len(rowdem[inicios[i]:fines[i]])
            else:
                pos0=pos1
                pos1+=len(rowdem[inicios[i]:fines[i]])
            demdef[j,inicios[i]:fines[i]]=syndata[pos0:pos1,0]
            cdef[j,inicios[i]:fines[i]]=sol[2*dem.shape[0]+par]
            par+=1
    return demdef,adef,bdef,cdef

def getabc_reg1(amps,dem,lama,lamb,lamc):
    numpar=0
    indices=[]
    for j in range(dem.shape[0]):
        rowdem=dem[j,:]
        inicios,fines=get_traces(rowdem)
        indices.append((inicios,fines))
        numpar+=len(inicios)
    numpar+=2*dem.shape[0]
    demcp=np.copy(dem)
    data=np.zeros((np.sum(~np.isnan(dem))+3*numpar-2,1))
    G=np.zeros((np.sum(~np.isnan(dem))+3*numpar-2,numpar))
    filas=[]
    par=0
    for j in range(dem.shape[0]):
        if j%100==0:
            print('Filas offs',j)
        rowamps=amps[j,:]
        rowdem=demcp[j,:]
        inicios,fines=indices[j]
        if len(inicios)<2:
            tam=len(inicios)
        else:
            tam=2
        for i in range(tam):
            psyn=np.cumsum(rowamps[inicios[i]:fines[i]])
            pdem=rowdem[inicios[i]:fines[i]]
            pdata=pdem
            if i==0 and j==0:
                pos0=0
                pos1=len(rowdem[inicios[i]:fines[i]])
            else:
                pos0=pos1
                pos1+=len(rowdem[inicios[i]:fines[i]])
            data[pos0:pos1,0]=pdata
            G[pos0:pos1,j]=psyn
            G[pos0:pos1,dem.shape[0]+j]=np.array(range(len(psyn)))+1
            G[pos0:pos1,2*dem.shape[0]+par]=1
            par+=1
    for i in range(dem.shape[0]-2):
        G[pos1+i,dem.shape[0]+i]=1*lama
        G[pos1+i,dem.shape[0]+i+1]=-2*lama
        G[pos1+i,dem.shape[0]+i+2]=1*lama
        G[pos1+i,2*dem.shape[0]+i]=1*lamb
        G[pos1+i,2*dem.shape[0]+i+1]=-2*lamb
        G[pos1+i,2*dem.shape[0]+i+2]=1*lamb
        
    for i in range(2*dem.shape[0],numpar-2):
        G[pos1+i,i]=1*lamc
        G[pos1+i,i+1]=-2*lamc
        G[pos1+i,i+2]=1*lamc
    
    print('Least squares')
    #print(G,data)
    #print(np.sum(np.isnan(G)),np.sum(np.isnan(data)))
    sol = np.linalg.lstsq(G, data, rcond=None)[0]
    print('Least squares finished')
    print(np.sum(np.isnan(G)))
    syndata=np.matmul(G,sol)
    print(syndata)
    par=0
    adef=np.ones(dem.shape)*np.nan
    bdef=np.ones(dem.shape)*np.nan
    cdef=np.ones(dem.shape)*np.nan
    demdef=np.ones(dem.shape)*np.nan
    for j in range(dem.shape[0]):
        inicios,fines=indices[j]
        rowdem=demcp[j,:]
        adef[j,:]=sol[j]
        bdef[j,:]=sol[dem.shape[0]+j]
        if len(inicios)<2:
            tam=len(inicios)
        else:
            tam=2
        for i in range(tam):
            if i==0 and j==0:
                pos0=0
                pos1=len(rowdem[inicios[i]:fines[i]])
            else:
                pos0=pos1
                pos1+=len(rowdem[inicios[i]:fines[i]])
            demdef[j,inicios[i]:fines[i]]=syndata[pos0:pos1,0]
            cdef[j,inicios[i]:fines[i]]=sol[2*dem.shape[0]+par]
            par+=1
    return demdef,adef,bdef,cdef

def get_amp(demdef,adef,bdef,cdef,minshade,maxshade,pos,case=0):
    amps=np.ones(demdef.shape)*np.nan
    demdef=np.copy(demdef)
    for j in range(demdef.shape[0]):
        rowdem=demdef[j,:]
        rowa=adef[j,:]
        rowb=bdef[j,:]
        rowc=cdef[j,:]
        inicios,fines=get_traces(rowdem)
        if len(inicios)<2:
            tam=len(inicios)
        else:
            tam=2
        for i in range(tam):
            pdem=rowdem[inicios[i]:fines[i]]
            pa=rowa[inicios[i]:fines[i]]
            pb=rowb[inicios[i]:fines[i]]
            pc=rowc[inicios[i]:fines[i]]
            pdif=pdem[1::]-pdem[0:-1]
            pamp=rowdem[inicios[i]:fines[i]]*0
            pamp[0]=pdem[0]-pc[0]
            pamp[1::]=pdif
            pamp=pamp/pa-pb/pa
            amps[j,inicios[i]:fines[i]]=pamp
    rows=pos[:,0]
    urows=dict()
    
    for p in pos:
        if amps[p[0],p[1]]<=maxshade:
            if str(p[0]) not in urows.keys():
                urows[str(p[0])]=p[1]-1
    if case==2:
        for p in pos:
            if amps[p[0],p[1]]<=maxshade:
                amps[p[0],p[1]]=np.random.randint(maxshade-5,maxshade)
    #print(urows)
    if case==3:
        for p in pos:
            if not str(p[0]) in urows.keys():
                continue
            elif p[1]<=urows[str(p[0])]:
                continue
            else:
                h=demdef[p[0],urows[str(p[0])]]+(maxshade*adef[p[0],p[1]]+bdef[p[0],p[1]])*(p[1]-urows[str(p[0])])
                if demdef[p[0],p[1]]<=h:
                    amps[p[0],p[1]]=np.random.randint(maxshade-5,maxshade)
    
    return amps

def getab_reg1(key,amps,grd,std,lamb=0,lim=True):
    grd=np.copy(grd)
    grd[np.abs(grd)>1]=np.nan
    
    pos=np.logical_and(~np.isnan(amps),~np.isnan(grd))
    G=np.zeros((np.sum(pos),2*amps.shape[0]))
    data=np.zeros((np.sum(pos),1))
    for i in range(amps.shape[0]):
        
        if i==0:
            pos0=0
            pos1=len(amps[i,:][pos[i,:]])
        else:
            pos0=pos1
            pos1+=len(amps[i,:][pos[i,:]])
        data[pos0:pos1,0]=grd[i,:][pos[i,:]]
        G[pos0:pos1,2*i]=np.ones(amps[i,:][pos[i,:]].shape)
        G[pos0:pos1,2*i+1]=amps[i,:][pos[i,:]]
    print(np.mean(data),np.max(data),np.min(data))
    sol = np.linalg.lstsq(G.T.dot(G) + lamb * np.identity(G.shape[1]), G.T.dot(data), rcond=None)[0]
    trsa=np.array([sol[2*i] for i in range(amps.shape[0])])
    trsb=np.array([sol[2*i+1] for i in range(amps.shape[0])])
    
    tracesa=trsa   
    tracesb=trsb
    
    return tracesa,tracesb

def getab1(key,amps,grd,std):
    #print('Inversion ab para',key,'con',amps.shape[0],'filas')
    tracesa=np.empty((amps.shape[0],1))
    tracesa[:]=np.nan
    tracesb=np.empty((amps.shape[0],1))
    tracesb[:]=np.nan
    for i in range(amps.shape[0]):
        #if i%100==0:
        #    print('Fila AB',i)
        a,b=bestab1(key,amps[i,:],grd[i,:],std[i,:])
        tracesa[i,0]=a
        tracesb[i,0]=b
    return tracesa,tracesb

def getabc(key,amps,dem,std,mask):
    print('Inversion abc para',key,'con',amps.shape[0],'filas')
    tracesa=np.empty((amps.shape[0],3805))
    tracesa[:]=np.nan
    tracesb=np.empty((amps.shape[0],3805))
    tracesb[:]=np.nan
    tracesc=np.empty((amps.shape[0],3805))
    tracesc[:]=np.nan
    for i in range(amps.shape[0]):
        if i%100==0:
            print('ABC para fila',i)
        samplesa,samplesb,samplesc=bestabc(key,amps[i,:],dem[i,:],std[i,:],mask[i,:])
        tracesa[i,0:len(samplesa)]=samplesa
        tracesb[i,0:len(samplesb)]=samplesb
        tracesc[i,0:len(samplesc)]=samplesc
    return tracesa,tracesb,tracesc

def getdemdef_reg(key,dem,grads,lamb,lim=True):
    numpar=0
    indices=[]
    for j in range(grads.shape[0]):
        rowgrads=grads[j,:]
        inicios,fines=get_traces(rowgrads)
        indices.append((inicios,fines))
        numpar+=len(inicios)
        
    demcp=np.copy(dem)
    data=np.zeros((np.sum(~np.isnan(grads))+numpar-2,1))
    G=np.zeros((np.sum(~np.isnan(grads))+numpar-2,numpar))
    filas=[]
    par=0
    for j in range(grads.shape[0]):
        if j%100==0:
            print('Filas offs',j)
        rowgrads=grads[j,:]
        rowdem=demcp[j,:]
        inicios,fines=indices[j]
        if len(inicios)<2 or not lim:
            tam=len(inicios)
        else:
            tam=2
        for i in range(tam):
            psyn=np.cumsum(rowgrads[inicios[i]:fines[i]])
            pdem=rowdem[inicios[i]:fines[i]]
            pdata=pdem-psyn
            if i==0 and j==0:
                pos0=0
                pos1=len(rowdem[inicios[i]:fines[i]])
            else:
                pos0=pos1
                pos1+=len(rowdem[inicios[i]:fines[i]])
            data[pos0:pos1,0]=pdata
            G[pos0:pos1,par]=1
            par+=1
    for i in range(numpar-2):
        G[pos1+i,i]=1*lamb
        G[pos1+i,i+1]=-2*lamb
        G[pos1+i,i+2]=1*lamb
    print('Least squares')
    #print(G,data)
    print(np.sum(np.isnan(G)),np.sum(np.isnan(data)))
    sol = np.linalg.lstsq(G, data, rcond=None)[0]
    #print(sol)
    print('Least squares finished')
    resm=np.linalg.norm(sol)
    
    par=0
    adef=np.ones(grads.shape)*np.nan
    demdef=np.ones(grads.shape)*np.nan
    for j in range(grads.shape[0]):
        rowgrads=grads[j,:]
        rowdem=demcp[j,:]
        inicios,fines=indices[j]
        if len(inicios)<2 or not lim:
            tam=len(inicios)
        else:
            tam=2
        for i in range(tam):
            psyn=np.cumsum(rowgrads[inicios[i]:fines[i]])
            adef[j,inicios[i]:fines[i]]=sol[par]
            #print(psyn,sol[par])
            demdef[j,inicios[i]:fines[i]]=psyn+sol[par]
            par+=1
    resd=np.nansum((demdef-dem)**2)
    return demdef,adef,resd,resm

def getdemdef1(key,dem,grads,std,lim=True):
    #grads=(amps.T*a).T+np.outer(c,np.ones(amps[0,:].shape))
    demcp=np.copy(dem)
    demdef=np.ones(grads.shape)*np.nan
    adef=np.ones(grads.shape)*np.nan
    filas=[]
    for j in range(grads.shape[0]):
        #if j%100==0:
        #    print('Filas offs',j)
        rowgrads=grads[j,:]
        rowdem=demcp[j,:]
        rowstd=std[j,:]
        inicios,fines=get_traces(rowgrads)
        if len(inicios)<2 or not lim:
            tam=len(inicios)
        else:
            tam=2
        for i in range(tam):
            psyn=np.cumsum(rowgrads[inicios[i]:fines[i]])
            pdem=rowdem[inicios[i]:fines[i]]
            pstd=rowstd[inicios[i]:fines[i]]
            b=bestoffs(key,psyn,pdem,pstd)
            rows=np.zeros((3,))
            rows[0]=j
            rows[1]=inicios[i]
            rows[2]=fines[i]
            adef[j,inicios[i]:fines[i]]=b
            filas.append(rows)
            demdef[j,inicios[i]:fines[i]]=psyn+b
    filas=np.array(filas)
    return demdef,filas,adef

def getdemdef2(key,dem,grads,lim=True):
    integration=integrate(grads)
    demcp=np.copy(dem)
    demdef=np.ones(grads.shape)*np.nan
    adef=np.ones(grads.shape)*np.nan
    filas=[]
    for j in range(integration.shape[0]):
        #if j%100==0:
        #    print('Filas offs',j)
        rowint=integration[j,:]
        rowdem=demcp[j,:]
        iniciosint,finesint=get_traces(rowint)
        #print(iniciosint,finesint)
        for i in range(len(iniciosint)):
            k=1
            valuedem=integration[j,finesint[i]-k]
            valuedemref=dem[j,finesint[i]-k]
            while(np.isnan(valuedemref) and iniciosint[i]<=finesint[i]-k):
                k+=1
                #valuedem=np.mean(integration[j,finesint[i]-k-4:finesint[i]-k])
                valuedem=integration[j,finesint[i]-k]
                #valuedemref=np.mean(dem[j,finesint[i]-k-4:finesint[i]-k])
                valuedemref=dem[j,finesint[i]-k]
            if np.isnan(valuedemref):
                iniciosdem,finesdem=get_traces(rowdem)
                middle=int((finesint[i]+iniciosint[i])/2)
                pos=np.argmin(np.abs(np.array(finesdem)-middle))
                valuedemref=dem[j,finesdem[pos]-1]
                valuedem=integration[j,middle]
            integration[j,iniciosint[i]:finesint[i]]+=valuedemref-valuedem
    return integration

def getdemdef(key,dem,amps,grd,std,a,c,mask):
    inicio=time.time()
    #print('Inversion dem para',key,'con',amps.shape[0],'filas')
    grad_syn=(amps.T*a).T+np.outer(c,np.ones(amps[0,:].shape))
    #grad_syn=(amps.T*a).T+np.outer(c,amps[0,:]*0+1)
    gradfilt=grd
    res=np.abs(grad_syn-gradfilt)
    copia=np.zeros(res.shape)
    copia[np.logical_and(res>=0.5,grad_syn>gradfilt)]=1
    grad_syn[np.logical_and(res>=0.5,grad_syn>gradfilt)]=np.nan
    demcp=np.copy(dem)
    gradcp=np.copy(grad_syn)
    gradcp[np.isnan(gradcp)]=gradfilt[np.isnan(gradcp)]
    demsyn=np.nancumsum(gradcp,axis=1)
    demsyn[np.isnan(gradcp)]=np.nan
    gradfiltcp=np.copy(gradfilt)
    gradfiltcp[np.isnan(demsyn)]=np.nan
    demfilt=np.nancumsum(gradfiltcp,axis=1)
    demdef=np.empty(demsyn.shape)
    demdef[:]=np.nan
    adef=np.empty(demsyn.shape)
    adef[:]=np.nan
    amin=np.empty(demsyn.shape)
    amin[:]=np.nan
    amax=np.empty(demsyn.shape)
    amax[:]=np.nan
    errdef=np.empty(demsyn.shape)
    errdef[:]=np.nan
    #print('Filas',demsyn.shape[0])
    filas=[]
    samples=[]
    for j in range(demsyn.shape[0]):
        if j%100==0:
            print('Fila off',j)
        rowsyn=demsyn[j,:]
        rowdem=demfilt[j,:]
        rowstd=std[j,:]
        rowmask=mask[j,:]
        posnan=np.argwhere(np.isnan(rowsyn))
        if len(posnan)>1:
            for i,pos in enumerate(posnan):
                pos=pos[0]
                if i==0 and not pos==0:
                    bsam,b,bmin,bmax=bestoffs(key,rowsyn[0:pos],rowdem[0:pos],rowstd[0:pos],rowmask[0:pos])
                    amin[j,0:pos]=bmin+demcp[j,0]
                    amax[j,0:pos]=bmax+demcp[j,0]
                    adef[j,0:pos]=b+demcp[j,0]
                    bsam+=demcp[j,0]
                    samples.append(bsam)
                    rows=[]
                    rows.append(j)
                    rows.append(0)
                    rows.append(pos)
                    rows=np.array(rows)
                    filas.append(rows)
                    demdef[j,0:pos]=rowsyn[0:pos]+b+demcp[j,0]
                    demfilt[j,0:pos]+=demcp[j,0]
                    
                elif i<len(posnan)-1 or pos==0:
                    if not posnan[i+1][0]==pos+1:
                        bsam,b,bmin,bmax=bestoffs(key,rowsyn[pos+1:posnan[i+1][0]],rowdem[pos+1:posnan[i+1][0]],rowstd[pos+1:posnan[i+1][0]],rowmask[pos+1:posnan[i+1][0]])
                        amin[j,pos+1:posnan[i+1][0]]=bmin+(demcp[j,pos+1]-demfilt[j,pos+1])
                        amax[j,pos+1:posnan[i+1][0]]=bmax+(demcp[j,pos+1]-demfilt[j,pos+1])
                        adef[j,pos+1:posnan[i+1][0]]=b+(demcp[j,pos+1]-demfilt[j,pos+1])
                        bsam+=(demcp[j,pos+1]-demfilt[j,pos+1])
                        samples.append(bsam)
                        rows=[]
                        rows.append(j)
                        rows.append(pos+1)
                        rows.append(posnan[i+1][0])
                        rows=np.array(rows)
                        filas.append(rows)
                        demdef[j,pos+1:posnan[i+1][0]]=rowsyn[pos+1:posnan[i+1][0]]+b+(demcp[j,pos+1]-demfilt[j,pos+1])
                        demfilt[j,pos+1:posnan[i+1][0]]+=(demcp[j,pos+1]-demfilt[j,pos+1])
                        
                else:
                    if pos+1<len(rowsyn):
                        bsam,b,bmin,bmax=bestoffs(key,rowsyn[pos+1::],rowdem[pos+1::],rowstd[pos+1::],rowmask[pos+1::])
                        amin[j,pos+1::]=bmin+(demcp[j,pos+1]-demfilt[j,pos+1])
                        amax[j,pos+1::]=bmax+(demcp[j,pos+1]-demfilt[j,pos+1])
                        adef[j,pos+1::]=b+(demcp[j,pos+1]-demfilt[j,pos+1])
                        bsam+=(demcp[j,pos+1]-demfilt[j,pos+1])
                        samples.append(bsam)
                        rows=[]
                        rows.append(j)
                        rows.append(pos+1)
                        rows.append(len(rowsyn))
                        rows=np.array(rows)
                        filas.append(rows)
                        demdef[j,pos+1::]=rowsyn[pos+1::]+b+(demcp[j,pos+1]-demfilt[j,pos+1])
                        demfilt[j,pos+1::]+=(demcp[j,pos+1]-demfilt[j,pos+1])
                        
                    else:
                        continue
        else:
            bsam,b,bmin,bmax=bestoffs(key,rowsyn,rowdem,rowstd,rowmask)
            amin[j,:]=bmin+demcp[j,0]
            amax[j,:]=bmax+demcp[j,0]
            adef[j,:]=b+demcp[j,0]
            bsam+=demcp[j,0]
            samples.append(bsam)
            rows=[]
            rows.append(j)
            rows.append(0)
            rows.append(len(rowsyn))
            rows=np.array(rows)
            filas.append(rows)
            demdef[j,:]=rowsyn+b+demcp[j,0]
            demfilt[j,:]+=demcp[j,0]
            
    print(len(demdef[j,:][~np.isnan(demdef[j,:])]),len(rowsyn[~np.isnan(rowsyn)]))
    demfilt[np.isnan(demdef)]=np.nan
    filas=np.array(filas)
    samples=np.array(samples)
    lens=[len(sample) for sample in samples]
    muestras=np.empty((samples.shape[0],int(np.max(lens))))
    muestras[:]=np.nan
    for i,sample in enumerate(samples):
        muestras[i,0:len(sample)]=sample
    fin=time.time()
    return demdef,muestras,filas

def getdem_spr(key,maskoutan=[],maskoutde=[],maskin=[],demref=[],h5file='descending.h5'):
    print('DEM for',key)
    h5i = h5py.File(h5file,'r')
    demcp=h5i[key+'/dem'][:]
    grdcp=h5i[key+'/grad'][:]
    ampcp=h5i[key+'/amps'][:]
    loncp=h5i[key+'/lon'][:]
    latcp=h5i[key+'/lat'][:]
    #stdcp=np.load('stdamps.npy')
    stdcp=np.ones(ampcp.shape)
    
    h5i.close()

    shade=get_shadow(ampcp,loncp,latcp)

    grdcp[grdcp>10]=np.nan

    fampcp=filtrar(ampcp)
    fgrdcp=filtrar(grdcp)
    
    

    tracesa,tracesb=getab1(key,fampcp,fgrdcp,stdcp)
    ma=np.nanmean(tracesa,axis=1)
    mb=np.nanmean(tracesb,axis=1)
    grads=(ampcp.T*ma).T+np.outer(mb,np.ones(fampcp[0,:].shape))

    ampcp2an=np.copy(ampcp)
    ampcp2de=np.copy(ampcp)
    ampcp3=np.copy(ampcp)

    if len(maskoutan)==0 or len(maskoutde)==0 or len(maskin)==0:
        signos=grads/np.abs(grads)
        outsignos=np.ones(grads.shape)
        outsignos[:,0:signos.shape[1]-1]=signos[:,0:signos.shape[1]-1]*signos[:,1:signos.shape[1]]
        if int(outsignos.shape[0])%2==1:
            length=int(outsignos.shape[0]/2)+1
        else:
            length=int(outsignos.shape[0]/2)
        ini=int(outsignos.shape[0]/2)
        for i in range(length):
            j=ini-i
            i=i+ini
            #print(j)
            posnegar=np.argwhere(outsignos[i,:]<0)
            posnegar=np.array([pos[0] for pos in posnegar])
            posnegab=np.argwhere(outsignos[j,:]<0)
            posnegab=np.array([pos[0] for pos in posnegab])
            if i==ini:
                pos0ar=posnegar[0]
                pos1ar=posnegar[-1]
                pos0ab=posnegab[0]
                pos1ab=posnegab[-1]
            else:
                if len(posnegab)>0:
                    min0=np.min(np.abs(posnegab-pos0ab))
                    min1=np.min(np.abs(posnegab-pos1ab))
                    if min0<100 and min1<100: 
                        poss0=np.abs(posnegab-pos0ab)
                        poss1=np.abs(posnegab-pos1ab)
                        if not np.argmin(poss0)==np.argmin(poss1):
                            pos0ab=posnegab[np.argmin(poss0)]
                            pos1ab=posnegab[np.argmin(poss1)]
                if len(posnegar)>0:
                    min0=np.min(np.abs(posnegar-pos0ar))
                    min1=np.min(np.abs(posnegar-pos1ar))
                    if min0<100 and min1<100: 
                        poss0=np.abs(posnegar-pos0ar)
                        poss1=np.abs(posnegar-pos1ar)
                        if not np.argmin(poss0)==np.argmin(poss1):
                            pos0ar=posnegar[np.argmin(poss0)]
                            pos1ar=posnegar[np.argmin(poss1)]
            ampcp2an[i,pos0ar::]=np.nan
            ampcp2de[i,0:pos1ar+1]=np.nan
            ampcp2an[j,pos0ab::]=np.nan
            ampcp2de[j,0:pos1ab+1]=np.nan
            ampcp3[i,0:pos0ar+1]=np.nan
            ampcp3[i,pos1ar::]=np.nan
            ampcp3[j,0:pos0ab+1]=np.nan
            ampcp3[j,pos1ab::]=np.nan
        maskin=np.logical_not(np.isnan(ampcp3))
        maskoutan=np.logical_not(np.isnan(ampcp2an))
        maskoutde=np.logical_not(np.isnan(ampcp2de))
    else:
        ampcp2an[~maskoutan]=np.nan
        ampcp2de[~maskoutde]=np.nan
        ampcp3[~maskin]=np.nan
    
    fampcp2an=filtrar(ampcp2an)
    tracesa,tracesb=getab1(key,fampcp2an,fgrdcp,stdcp)
    mapos=np.nanmean(tracesa,axis=1)
    mbpos=np.nanmean(tracesb,axis=1)
    gradscp2an=(ampcp2an.T*mapos).T+np.outer(mbpos,np.ones(fampcp2an[0,:].shape))
    
    fampcp2de=filtrar(ampcp2de)
    tracesa,tracesb=getab1(key,fampcp2de,fgrdcp,stdcp)
    maneg=np.nanmean(tracesa,axis=1)
    mbneg=np.nanmean(tracesb,axis=1)
    gradscp2de=(ampcp2de.T*maneg).T+np.outer(mbneg,np.ones(fampcp2de[0,:].shape))
    
    gradscp2=np.copy(gradscp2de)
    gradscp2[~np.isnan(gradscp2an)]=gradscp2an[~np.isnan(gradscp2an)]
    
    
    #gradscp2=(ampcp2.T*ma).T+np.outer(mb,np.ones(fampcp[0,:].shape))
    #gradscp2[ampcp2<=int(shade)]=grdcp[ampcp2<=int(shade)]

    demdef,filas,adef=getdemdef1(key,demcp,gradscp2,stdcp)
    if len(demref)==0 or int(key)>20200107:
        fakedem=np.ones(demdef.shape)*np.nan
        for i in range(fakedem.shape[0]):
            inicios,fines=get_traces(demdef[i,:])
            alt1=demdef[i,fines[0]-1]
            alt2=demdef[i,inicios[-1]+1]
            slope=(alt2-alt1)/(inicios[-1]-fines[0]+2)
            offset=alt2-slope*(inicios[-1]+1)
            fakedem[i,:]=slope*np.array(range(fakedem.shape[1]))+offset
        
    
    ellipse=get_ellipse(loncp,latcp,xcen=-163.971,ycen=54.7562,ax=None,ay=None)
    gradscp3=(ampcp3.T*ma).T+np.outer(mb,np.ones(fampcp[0,:].shape))
    ampcp3pos=np.copy(ampcp3)
    ampcp3pos[gradscp3<=0]=np.nan
    ampcp3neg=np.copy(ampcp3)
    ampcp3neg[gradscp3>0]=np.nan
    #ampcp3neg[ampcp3<int(shade)+1]=np.nan
    gradscp3pos=(ampcp3pos.T*mapos).T+np.outer(mbpos,np.ones(fampcp[0,:].shape))
    gradscp3neg=(ampcp3neg.T*ma).T+np.outer(mb,np.ones(fampcp[0,:].shape))
    gradscp3=np.copy(gradscp3pos)
    gradscp3[~np.isnan(gradscp3neg)]=gradscp3neg[~np.isnan(gradscp3neg)]
    #print('Porcentaje de shade',len(gradscp3[ampcp3<int(shade)+1]),len(gradscp3[maskin==1]))
    
    gradscp3[ampcp3<int(shade)+1]=0
    positivos=np.ones(gradscp3.shape)*np.nan
    positivos[np.isnan(gradscp3)]=1
    #positivos[np.logical_and(np.isnan(ellipse),gradscp3>0)]=1
    #gradscp3[np.logical_and(np.isnan(ellipse),gradscp3>0)]=np.nan
    positivos[np.logical_and(maskin,gradscp3>0)]=1
    gradscp3[np.logical_and(maskin,gradscp3>0)]=np.nan
    
    

    if len(demref)==0:
        demdef1,filas1,adef=getdemdef1(key,fakedem,gradscp3,stdcp,lim=False)
    else:
        if int(key)>20200107:
            demdef1,filas,adef=getdemdef1(key,fakedem,gradscp3,stdcp,lim=False)
        else:
            demdef1=getdemdef2(key,demref,gradscp3,lim=False)
        #demdef1=lsquares.getdemdef2(key,fakedem,gradscp3,lim=False)
    #demdef1=lsquares.integrate(gradscp3)
    
    demdef1[ampcp3<int(shade)+1]=np.nan

    demdef2=np.copy(demdef1)
    '''
    for i in range(demdef.shape[0]):
        inicios,fines=lsquares.get_traces(demdef[i,:])
        inicios1,fines1=lsquares.get_traces(demdef1[i,:])
        inicio=fines[0]
        fin=inicios[1]
        
        inicio1=inicios1[0]
        fin1=fines1[-1]
        off0=demdef[i,inicio-1]-demdef1[i,inicio1+1]
        #dif=np.max(demcp[np.isnan(ellipse)])-np.sort(demcp[np.isnan(ellipse)])[9]
        off1=demdef[i,fin+1]-demdef1[i,fin1-1]
        if len(inicios1)>1:
            demdef2[i,inicios1[0]:fines1[0]+1]+=off0
            demdef2[i,inicios1[-1]:fines1[-1]+1]+=off0
            off=(off0+off1)/2
            for j in range(1,len(inicios1)-1):
                demdef2[i,inicios1[j]:fines1[j]+1]+=off0
        else:
            demdef2[i,:]+=off0
    '''
    demdef3=np.copy(demdef)
    demdef3[~np.isnan(demdef2)]=demdef2[~np.isnan(demdef2)]
    
    return demdef3,gradscp3,ampcp,maskoutan,maskoutde,maskin,positivos,shade*ma+mb