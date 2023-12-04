from shutil import copyfile, move # Utilities for copying and moving files
from osgeo import gdal            # GDAL support for reading virtual files
import os                         # To create and remove directories
import matplotlib.pyplot as plt   # For plotting
import numpy as np                # Matrix calculations
import glob                       # Retrieving list of files
import boto3                      # For talking to s3 bucket
import getpass
import subprocess
from scipy.ndimage import spline_filter
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from skimage.restoration import denoise_nl_means, estimate_sigma
import h5py
import multiprocessing
from functools import partial

# directory in which the notebook resides
if 'tutorial_home_dir' not in globals():
    tutorial_home_dir = os.getcwd()
#print("Notebook directory: ", tutorial_home_dir)

# directory for data downloads
slc_dir = os.path.join(tutorial_home_dir,'data', 'slcs')
orbit_dir = os.path.join(tutorial_home_dir, 'data', 'orbits')
insar_dir = os.path.join(tutorial_home_dir, 'insar')


# defining backup dirs in case of download issues on the local server
s3 = boto3.resource("s3")
data_backup_bucket = s3.Bucket("asf-jupyter-data")
data_backup_dir = "STRP"

# generate all the folders in case they do not exist yet
os.makedirs(slc_dir, exist_ok=True)
os.makedirs(orbit_dir, exist_ok=True)
os.makedirs(insar_dir, exist_ok=True)

# Always start at the notebook directory    
os.chdir(tutorial_home_dir)

def calc_std(stack):
    fd = h5py.File(stack, mode="r")
    scenes = list(fd.keys())
    r, c = fd[scenes[0]]["amps"].shape
    cube = np.zeros((len(scenes), r, c))
    
    for i, key in enumerate(fd.keys()):
        cube[i,:,:] = fd[key]["amps"][:]
        
    fd.close()
    
    std = np.std(cube, axis=0)
    
    np.save("stdamps.npy", std)

def preprocessing(lons,lats,projections='projections.h5',path='D'):
    if path=='D':
        output='descending.h5'
    else:
        output='ascending.h5'
    h5i = h5py.File(projections,'r')
    #lons=[-163.9814,-163.9614]
    #lats=[54.7554,54.7570]
    sigma_amp=150
    sigma_dem=0.5
    patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=5,  # 13x13 search area
                    multichannel=True)
    keys=[key for key in h5i]
    keys=[keys[0]]+keys
    llaves=keys
    h5i.close()
    for i,key in enumerate(llaves):
        h5i = h5py.File(projections,'r')
        lonrdr=h5i[key+'/lon'][:]
        latrdr=h5i[key+'/lat'][:]
        x0,y0,xsizet,ysizet=get_box2(lonrdr,latrdr,lons=lons,lats=lats,path=path)
        if i==0:
            amps=h5i[key+'/amps'][:]
            llave=key.split('_')[0]
        else:
            amps=h5i[key+'/amps1'][:]
            llave=key.split('_')[1]
        ampscp=np.copy(amps[y0:y0+ysizet,x0:x0+xsizet])

        ampscp[np.isnan(ampscp)]=0

        dem=h5i[key+'/dem'][:]
        h5i.close()
        print('Preprocessing for ',llave)
        grd=denoise_nl_means(np.gradient(dem,axis=1)[y0:y0+ysizet,x0:x0+xsizet], h=0.6 * sigma_dem, sigma=sigma_dem, fast_mode=True, **patch_kw)
        mask=get_mask(lonrdr,path=path)
        lonrdr=lonrdr*mask

        ampscp1=denoise_nl_means(ampscp, h=0.6 * sigma_amp, sigma=sigma_amp, fast_mode=True, **patch_kw)
        ampscp=gaussian_filter(ampscp1, sigma=1)
        ampscp=ampscp*mask[y0:y0+ysizet,x0:x0+xsizet]
        grd=grd*mask[y0:y0+ysizet,x0:x0+xsizet]


        dem1=(dem*mask)[y0:y0+ysizet,x0:x0+xsizet]
        dem=dem1

        loncp=(lonrdr*mask)[y0:y0+ysizet,x0:x0+xsizet]
        latcp=(latrdr*mask)[y0:y0+ysizet,x0:x0+xsizet]

        ampscp[grd<-1]=np.nan
        dem[grd<-1]=np.nan
        loncp[grd<-1]=np.nan
        latcp[grd<-1]=np.nan
        grd[grd<-1]=np.nan

        if os.path.exists(output):
            h5o=h5py.File(output,'a')
        else:
            h5o=h5py.File(output,'w')
        grp=h5o.create_group(llave)
        grp.create_dataset('amps', data=ampscp, compression="gzip")
        grp.create_dataset('grad', data=grd, compression="gzip")
        grp.create_dataset('dem', data=dem, compression="gzip")
        grp.create_dataset('lon', data=loncp, compression="gzip")
        grp.create_dataset('lat', data=latcp, compression="gzip")
        h5o.close()

def coregistration(output='projections.h5',index=1):
    archivos=sorted(glob.glob('./data/slcs1/*'))
    for i in range(len(archivos[index::])):
        i+=index
        basename1=os.path.basename(archivos[0])
        fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
        file1='../data/slcs1/'+basename1+'/'+basename1+'.xml'
        basename2=os.path.basename(archivos[i])
        fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
        file2='../data/slcs1/'+basename2+'/'+basename2+'.xml'
        try:
            print('insar of ',file1,file2)
            insar(file1,file2)
            demrdr=readdata('./temporal/geometry/z.rdr.full')
            lonrdr=readdata('./temporal/geometry/lon.rdr.full')
            latrdr=readdata('./temporal/geometry/lat.rdr.full')
            amps,angles=readcomplexdata('./temporal/'+fecha1+'_slc/'+fecha1+'.slc.vrt')
            amps1,angles1=readcomplexdata('./temporal/coregisteredSlc/coarse_coreg.slc.vrt')
            if not os.path.exists('./'+output):
                h5f = h5py.File('./'+output, 'w')
            else:
                h5f = h5py.File('./'+output, 'a')
            grp=h5f.create_group(fecha1+'_'+fecha2)
            grp.create_dataset('dem', data=demrdr, compression="gzip")
            grp.create_dataset('lon', data=lonrdr, compression="gzip")
            grp.create_dataset('lat', data=latrdr, compression="gzip")
            grp.create_dataset('amps', data=amps, compression="gzip")
            grp.create_dataset('amps1', data=amps1, compression="gzip")
            subprocess.call('rm -rf temporal', stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,shell=True)
            h5f.close()
        except:
            continue

def menores_atras(row,path='D'):
    atras=10000
    nuevo=np.empty(row.shape)
    nuevo[0]=row[0]
    atras=row[0]
    
    for i,elem in enumerate(row[0:len(row)-1]):
        condicion=False
        if path=='D' and atras>=elem-0.0001:
            condicion=True
        elif path=='A' and atras<=elem+0.0001:
            condicion=True
        if condicion:
            nuevo[i+1]=1
            atras=elem
        else:
            nuevo[i+1]=np.nan
    return nuevo

def get_mask(lons,path='D'):
    mask=np.empty(lons.shape)
    for k in range(lons.shape[0]):
        mask[k,:]=menores_atras(lons[k,:],path)
    return mask

def get_ellipse(x,y,xcen,ycen,ax,ay):
    elipse=np.ones(x.shape)
    elipse[np.power(x-xcen,2)/np.power(ax,2)+np.power(y-ycen,2)/np.power(ay,2)<=1]=np.nan
    elipse[np.logical_and(x<xcen,np.power(y-ycen,2)<np.power(ay,2))]=np.nan
    return elipse

def get_box2(lonrdr,latrdr,lons=[-163.977,-163.967],lats=[54.7535,54.7585],path='D'):
    mask=get_mask(lonrdr,path)
    loncp=np.copy(lonrdr)*mask
    latcp=np.copy(latrdr)*mask
    cond1=np.logical_and(loncp>=lons[0],loncp<=lons[1])
    cond2=np.logical_and(latcp>=lats[0],latcp<=lats[1])
    cond3=np.logical_and(cond1,cond2)
    cond4=np.logical_not(np.isnan(mask))
    cond=np.logical_and(cond3,cond4)
    maxi=0
    mini=100000
    for i in range(int(cond.shape[1]-1)):
        f=cond.shape[1]-1-i
        if np.sum(cond[:,i]*cond[:,i+1])>0 and mini==100000:
            mini=i
        if np.sum(cond[:,f]*cond[:,f-1])>0 and maxi==0:
            maxi=f
        if not maxi==0 and not mini==100000:
            break
    x0=mini
    xf=maxi
    xsize=xf-x0
    maxi=0
    mini=100000
    for i in range(int(cond.shape[0]-1)):
        f=cond.shape[0]-1-i
        if np.sum(cond[i,:]*cond[i+1,:])>0 and mini==100000:
            mini=i
        if np.sum(cond[f,:]*cond[f-1,:])>0 and maxi==0:
            maxi=f
        if not maxi==0 and not mini==100000:
            break
    y0=mini
    yf=maxi
    ysize=yf-y0
    #print(x0,y0,xsize,ysize)
    return x0,y0,xsize,ysize            

def coregistration1(index=1):
    archivos=sorted(glob.glob('./data/slcs1/*'))
    for i in range(len(archivos[index::])):
        i+=index
        basename1=os.path.basename(archivos[0])
        fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
        file1='../data/slcs1/'+basename1+'/'+basename1+'.xml'
        basename2=os.path.basename(archivos[i])
        fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
        file2='../data/slcs1/'+basename2+'/'+basename2+'.xml'
        try:
            insar(file1,file2)
        except:
            continue
            
def GetExtent(raster):
    """ Return list of corner coordinates from a gdal Dataset """
    ds=gdal.Open(raster)
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    return xmin,xmax,ymin,ymax

# Utility to plot a 2D array
def readdata(GDALfilename, band=1, background=None,
             datamin=None, datamax=None,
             interpolation='nearest',
             nodata = None):
    
    # Read the data into an array
    ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
    data = ds.GetRasterBand(band).ReadAsArray()
    transform = ds.GetGeoTransform()
    ds = None
    
    try:
        if nodata is not None:
            data[data == nodata] = np.nan
    except:
        pass

    # put all zero values to nan and do not plot nan
    if background is None:
        try:
            data[data==0]=np.nan
        except:
            pass
    
    return data

# Utility to plot interferograms
def readcomplexdata(GDALfilename):
    # Load the data into numpy array
    ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
    slc = ds.GetRasterBand(1).ReadAsArray()
    transform = ds.GetGeoTransform()
    ds = None
    
    # put all zero values to nan and do not plot nan
    try:
        slc[slc==0]=np.nan
    except:
        pass

    return np.abs(slc),np.angle(slc)

# Utility to plot a 2D array
def plotdata(GDALfilename, band=1,
             title=None,colormap='gray',
             aspect=1, background=None,
             datamin=None, datamax=None,
             interpolation='nearest',
             nodata = None,
             draw_colorbar=True, colorbar_orientation="horizontal"):
    
    # Read the data into an array
    ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
    data = ds.GetRasterBand(band).ReadAsArray()
    transform = ds.GetGeoTransform()
    ds = None
    
    try:
        if nodata is not None:
            data[data == nodata] = np.nan
    except:
        pass
        
    # getting the min max of the axes
    firstx = transform[0]
    firsty = transform[3]
    deltay = transform[5]
    deltax = transform[1]
    lastx = firstx+data.shape[1]*deltax
    lasty = firsty+data.shape[0]*deltay
    ymin = np.min([lasty,firsty])
    ymax = np.max([lasty,firsty])
    xmin = np.min([lastx,firstx])
    xmax = np.max([lastx,firstx])

    # put all zero values to nan and do not plot nan
    if background is None:
        try:
            data[data==0]=np.nan
        except:
            pass
    
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)
    cax = ax.imshow(data, vmin = datamin, vmax=datamax,
                    cmap=colormap, extent=[xmin,xmax,ymin,ymax],
                    interpolation=interpolation)
    ax.set_title(title)
    if draw_colorbar is not None:
        cbar = fig.colorbar(cax,orientation=colorbar_orientation)
    ax.set_aspect(aspect)    
    plt.show()
    
    # clearing the data
    data = None

# Utility to plot interferograms
def plotcomplexdata(GDALfilename,
                    title=None, aspect=1,
                    datamin=None, datamax=None,
                    interpolation='nearest',
                    draw_colorbar=None, colorbar_orientation="horizontal"):
    # Load the data into numpy array
    ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
    slc = ds.GetRasterBand(1).ReadAsArray()
    transform = ds.GetGeoTransform()
    ds = None
    
    # getting the min max of the axes
    firstx = transform[0]
    firsty = transform[3]
    deltay = transform[5]
    deltax = transform[1]
    lastx = firstx+slc.shape[1]*deltax
    lasty = firsty+slc.shape[0]*deltay
    ymin = np.min([lasty,firsty])
    ymax = np.max([lasty,firsty])
    xmin = np.min([lastx,firstx])
    xmax = np.max([lastx,firstx])

    # put all zero values to nan and do not plot nan
    try:
        slc[slc==0]=np.nan
    except:
        pass

    
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(1,2,1)
    cax1=ax.imshow(np.abs(slc), vmin = datamin, vmax=datamax,
                   cmap='gray', extent=[xmin,xmax,ymin,ymax],
                   interpolation=interpolation)
    ax.set_title(title + " (amplitude)")
    if draw_colorbar is not None:
        cbar1 = fig.colorbar(cax1,orientation=colorbar_orientation)
    ax.set_aspect(aspect)

    ax = fig.add_subplot(1,2,2)
    cax2 =ax.imshow(np.angle(slc), cmap='rainbow',
                    vmin=-np.pi, vmax=np.pi,
                    extent=[xmin,xmax,ymin,ymax],
                    interpolation=interpolation)
    ax.set_title(title + " (phase [rad])")
    if draw_colorbar is not None:
        cbar2 = fig.colorbar(cax2, orientation=colorbar_orientation)
    ax.set_aspect(aspect)
    plt.show()
    
    # clearing the data
    slc = None

# Utility to plot multiple similar arrays
def plotstackdata(GDALfilename_wildcard, band=1,
                  title=None, colormap='gray',
                  aspect=1, datamin=None, datamax=None,
                  interpolation='nearest',
                  draw_colorbar=True, colorbar_orientation="horizontal"):
    # get a list of all files matching the filename wildcard criteria
    GDALfilenames = glob.glob(GDALfilename_wildcard)
    
    # initialize empty numpy array
    data = None
    for GDALfilename in GDALfilenames:
        ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
        data_temp = ds.GetRasterBand(band).ReadAsArray()   
        ds = None
        
        if data is None:
            data = data_temp
        else:
            data = np.vstack((data,data_temp))

    # put all zero values to nan and do not plot nan
    try:
        data[data==0]=np.nan
    except:
        pass            
            
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)
    cax = ax.imshow(data, vmin = datamin, vmax=datamax,
                    cmap=colormap, interpolation=interpolation)
    ax.set_title(title)
    if draw_colorbar is not None:
        cbar = fig.colorbar(cax,orientation=colorbar_orientation)
    ax.set_aspect(aspect)    
    plt.show() 

    # clearing the data
    data = None

# Utility to plot multiple simple complex arrays
def plotstackcomplexdata(GDALfilename_wildcard,
                         title=None, aspect=1,
                         datamin=None, datamax=None,
                         interpolation='nearest',
                         draw_colorbar=True, colorbar_orientation="horizontal"):
    # get a list of all files matching the filename wildcard criteria
    GDALfilenames = glob.glob(GDALfilename_wildcard)
    print(GDALfilenames)
    # initialize empty numpy array
    data = None
    for GDALfilename in GDALfilenames:
        ds = gdal.Open(GDALfilename, gdal.GA_ReadOnly)
        data_temp = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        
        if data is None:
            data = data_temp
        else:
            data = np.vstack((data,data_temp))

    # put all zero values to nan and do not plot nan
    try:
        data[data==0]=np.nan
    except:
        pass              
            
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(1,2,1)
    cax1=ax.imshow(np.abs(data), vmin=datamin, vmax=datamax,
                   cmap='gray', interpolation='nearest')
    ax.set_title(title + " (amplitude)")
    if draw_colorbar is not None:
        cbar1 = fig.colorbar(cax1,orientation=colorbar_orientation)
    ax.set_aspect(aspect)

    ax = fig.add_subplot(1,2,2)
    cax2 =ax.imshow(np.angle(data), cmap='rainbow',
                            interpolation='nearest')
    ax.set_title(title + " (phase [rad])")
    if draw_colorbar is not None:
        cbar2 = fig.colorbar(cax2,orientation=colorbar_orientation)
    ax.set_aspect(aspect)
    plt.show() 
    
    # clearing the data
    data = None

def residualsarc1(x0,grd,sar):
    a,d=x0
    sars=a*np.abs(grd)+d
    res=np.sum(np.power(sars[~np.isnan(sar)]-sar[~np.isnan(sar)],2))
    return res    

def residualsarc(x0,grd,sar):
    a,d=x0
    sars=a*grd+d
    res=np.nansum(np.power(sars-sar,2))
    return res

def residualsar(x0,grd,sar,los):
    a,b,c=x0
    sars=a*grd+b*los+c
    res=np.sum(np.power(sars[~np.isnan(sar)]-sar[~np.isnan(sar)],2))
    return res

def sartodif1(amps,amps1,grad,x0=0,y0=0,xsize=0,ysize=0):
    if xsize==0:
        xsize=amps.shape[1]
    if ysize==0:
        ysize=amps.shape[0]
    gradcp=np.copy(grad[y0:y0+ysize,x0:x0+xsize])
    gradfilt=gaussian_filter(gradcp, sigma=10)
    
    ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
    ampscp[np.isnan(ampscp)]=0
    ampsfilt=gaussian_filter(ampscp, sigma=5)
    
    result = minimize(residualsarc,x0=np.array([1.0,0.0]),args=(gradfilt,ampsfilt), method='Nelder-Mead', tol=1e-6)
    a,d=result.x
    
    qgrad=(ampsfilt-d)/a
    
    ampscp1=np.copy(amps1[y0:y0+ysize,x0:x0+xsize])
    ampscp1[np.isnan(ampscp1)]=0
    ampsfilt1=gaussian_filter(ampscp1, sigma=5)
    
    result = minimize(residualsarc,x0=np.array([1.0,0.0]),args=(gradfilt,ampsfilt1), method='Nelder-Mead', tol=1e-6)
    a1,d1=result.x
    
    qgrad1=(ampsfilt1-d1)/a1
    
    dif=(ampsfilt1-ampsfilt)
    demdif=np.cumsum((dif-(d1-d))/(a1-a),axis=1)
    residualcol=np.array([np.sum(np.abs(demdif[:,i])) for i in range(demdif.shape[1])])
    pos=np.argmin(residualcol)

    for i in range(demdif.shape[0]):
        offset=demdif[i,pos]
        demdif[i,:]+=offset

    for i in range(demdif.shape[1]):
        demdif[:,i]=gaussian_filter1d(demdif[:,i], sigma=1)
    
    return gradcp,gradfilt,ampscp,ampscp1,ampsfilt,ampsfilt1,qgrad,qgrad1,demdif

def sartodif12(amps,amps1,demrdr,x0=0,y0=0,xsize=0,ysize=0,row=None,col=None,off=None):
    if xsize==0:
        xsize=amps.shape[1]
    if ysize==0:
        ysize=amps.shape[0]
        
    demrdrcp=np.copy(demrdr[y0:y0+ysize,x0:x0+xsize])
    gradcp=np.gradient(demrdrcp,axis=1)
    
    sigma_amp=150
    sigma_dem=0.5
    
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
    
    
    gradfilt=denoise_nl_means(gradcp, h=0.6 * sigma_dem, sigma=sigma_dem,
                                 fast_mode=True, **patch_kw)
    
    ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
    ampscp[np.isnan(ampscp)]=0
    ampsfilt=denoise_nl_means(ampscp, h=0.6 * sigma_amp, sigma=sigma_amp,
                                 fast_mode=True, **patch_kw)
    
    result = minimize(residualsarc,x0=np.array([1.0,0.0]),args=(gradfilt,ampsfilt), method='Nelder-Mead', tol=1e-6)
    a,d=result.x
    
    qgrad=(ampsfilt-d)/a
    
    ampscp1=np.copy(amps1[y0:y0+ysize,x0:x0+xsize])
    ampscp1[np.isnan(ampscp1)]=0
    ampsfilt1=denoise_nl_means(ampscp1, h=0.6 * sigma_amp, sigma=sigma_amp,
                                 fast_mode=True, **patch_kw)
    
    result = minimize(residualsarc,x0=np.array([1.0,0.0]),args=(gradfilt,ampsfilt1), method='Nelder-Mead', tol=1e-6)
    a1,d1=result.x
    
    qgrad1=(ampsfilt1-d1)/a1
    
    dif=qgrad1-qgrad
    
    filt,coh,std=coherence(dif,win=10,thres=1)
    
    #filt[np.abs(filt)>0.6]=0
    if not off==None:
        filt-=off
    else:
        off=getmean(dif)
        filt-=off
    
    demdif,x,y=integrate(filt,row=row,col=col,std=std)
    filt[filt==-1*off]=0
    filt[np.isnan(filt)]=0
    
    return gradcp,gradfilt,ampscp,ampscp1,ampsfilt,ampsfilt1,qgrad,qgrad1,dif,filt,coh,std,demdif

def residualcol(x0,colamp,colgrd):
    a,b=x0
    colamps=a*colgrd+b
    colampcp=np.copy(colamp)
    #colampcp[colampcp-np.nanmean(colampcp)>np.nanstd(colampcp)]=np.nan
    res=np.sum(np.power(colamps[~np.isnan(colampcp)]-colamp[~np.isnan(colampcp)],2))
    return res

def bestpar(aes,bes,colamp,colgrd):
    aescp=np.copy(aes)
    bescp=np.copy(bes)
    colamps=np.outer(colgrd,aescp)+np.outer((colgrd*0+1),bescp)
    colampr=np.outer(colamp,(aescp*0+1))
    res=np.sum(np.power(colamps-colampr,2),axis=0)
    pos=np.argmin(res)
    return aescp[pos],bescp[pos]

def residualrow(b,rowsyn,rowdem):
    rowdems=rowsyn+b[0]
    res=np.nansum(np.power(rowdems-rowdem,2))
    return res

def bestoff(rowsyn,rowdem,b=0.0):
    result = minimize(residualrow,x0=np.array([0.0]),args=(rowsyn,rowdem), method='Nelder-Mead', tol=1e-6)
    b=result.x
    return b[0]

def sar2grad(amps,demrdr,x0,y0,xsize,ysize):
    if xsize==0:
        xsize=amps.shape[1]
    if ysize==0:
        ysize=amps.shape[0]
        
    demrdrcp=np.copy(demrdr[y0:y0+ysize,x0:x0+xsize])
    #demrdrcp=np.copy(demrdr[:,x0:x0+xsize])
    gradcp=np.gradient(demrdrcp,axis=1)
    
    sigma_amp=150
    sigma_dem=0.5
    
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
    
    
    gradfilt=denoise_nl_means(gradcp, h=0.6 * sigma_dem, sigma=sigma_dem,
                                 fast_mode=True, **patch_kw)
    
    ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
    #ampscp=np.copy(amps[:,x0:x0+xsize])
    ampscp[np.isnan(ampscp)]=0
    ampsfilt=denoise_nl_means(ampscp, h=0.6 * sigma_amp, sigma=sigma_amp,
                                 fast_mode=True, **patch_kw)
    
    grad_syn=np.zeros(gradcp.shape)
    count_syn=np.zeros(gradcp.shape)
    aes=[]
    bes=[]
    res=[]
    for j in range(grad_syn.shape[1]):
        if j<grad_syn.shape[1]-1:
            lim=1
        else:
            lim=grad_syn.shape[1]-j
        colamp=np.copy(ampsfilt[:,j:j+lim])
        colgrd=np.copy(gradfilt[:,j:j+lim])
        result = minimize(residualcol,x0=np.array([1.0,0.0]),args=(colamp,colgrd), method='Nelder-Mead', tol=1e-6)
        a,b=result.x
        #count_syn[:,j:j+lim]+=1
        aes.append(a)
        bes.append(b)
        res.append(np.sum(np.abs(((colamp-b)/a)-colgrd)))
        
    bestaes=[aes[i] for i in range(len(aes)) if res[i]<=20]
    bestbes=[bes[i] for i in range(len(bes)) if res[i]<=20]
    for j in range(grad_syn.shape[1]):
        colamp=np.copy(ampsfilt[:,j])
        colgrd=np.copy(gradfilt[:,j])
        a,b=bestpar(bestaes,bestbes,colamp,colgrd)
        grad_syn[:,j]=((colamp-b)/a)
        
    for j in range(grad_syn.shape[0]):
        rowamp=np.copy(ampsfilt[j,:])
        rowgrd=np.copy(gradfilt[j,:])
        a,b=bestpar(bestaes,bestbes,rowamp,rowgrd)
        grad_syn[j,:]=((rowamp-b)/a)
        
    return ampsfilt,gradfilt,grad_syn,aes,bes,res

def sar2grad2(amps,demrdr,x0,y0,xsize,ysize):
    if xsize==0:
        xsize=amps.shape[1]
    if ysize==0:
        ysize=amps.shape[0]
        
    demrdrcp=np.copy(demrdr[y0:y0+ysize,x0:x0+xsize])
    #demrdrcp=np.copy(demrdr[:,x0:x0+xsize])
    gradcp=np.gradient(demrdrcp,axis=1)
    
    sigma_amp=150
    sigma_dem=0.5
    
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
    
    
    #gradfilt=denoise_nl_means(gradcp, h=0.6 * sigma_dem, sigma=sigma_dem,fast_mode=True, **patch_kw)
    gradfilt=gaussian_filter(gradcp, sigma=10)
    
    ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
    #ampscp=np.copy(amps[:,x0:x0+xsize])
    ampscp[np.isnan(ampscp)]=0
    #ampsfilt=denoise_nl_means(ampscp, h=0.6 * sigma_amp, sigma=sigma_amp,fast_mode=True, **patch_kw)
    ampsfilt=gaussian_filter(ampscp, sigma=10)
    
    grad_syn=np.zeros(ampsfilt.shape)
    
    gradfiltcp=np.copy(gradfilt)
    #gradfiltcp[ampsfilt>150]=np.nan
    
    ampsfiltcp=np.copy(ampsfilt)
    #ampsfiltcp[ampsfilt>150]=np.nan
    
    #result = minimize(residualsarc,x0=np.array([1.0,0.0]),args=(gradfiltcp,ampsfiltcp), method='Nelder-Mead', tol=1e-6)
    #a1,b1=result.x
    
    
    #grad_syn[ampsfilt>150]=(ampsfilt[ampsfilt>150]-b1)/a1
    
    #gradfiltcp=np.copy(gradfilt)
    #gradfiltcp[ampsfilt<=150]=np.nan
    
    #ampsfiltcp=np.copy(ampsfilt)
    #ampsfiltcp[ampsfilt<=150]=np.nan
    
    
    result = minimize(residualsarc,x0=np.array([1.0,0.0]),args=(gradfiltcp,ampsfiltcp), method='Nelder-Mead', tol=1e-6)
    a1,b1=result.x
    
    #grad_syn[ampsfilt<=150]=(ampsfilt[ampsfilt<=150]-b1)/a1
    grad_syn=(ampsfilt-b1)/a1
    
    
    res=np.abs(gradfilt-grad_syn)
    
    grad_syncp=np.copy(grad_syn)
    grad_syncp[res<1.0]=np.nan
    gradfiltcp[res<1.0]=np.nan
    result = minimize(residualsarc,x0=np.array([1.0,0.0]),args=(gradfiltcp,grad_syncp), method='Nelder-Mead', tol=1e-6)
    a1,b1=result.x
    
    grad_syn[res>=1.0]=(grad_syncp[res>=1.0]-b1)/a1
        
    return ampsfilt,gradfilt,grad_syn

def getmean(dif,win=100):
    rows,cols=dif.shape
    rspaces=rows-win
    cspaces=cols-win
    minimo=1000
    for i in range(rspaces):
        for j in range(cspaces):
            std=np.nanstd(dif[i:i+win,j:j+win])
            if std<minimo:
                minimo=std
                mean=np.nanmean(dif[i:i+win,j:j+win])
    return mean

def sartodif13(amps,amps1,demrdr,x0=0,y0=0,xsize=0,ysize=0,row=None,col=None,off=None):
    if xsize==0:
        xsize=amps.shape[1]
    if ysize==0:
        ysize=amps.shape[0]
        
    demrdrcp=np.copy(demrdr[y0:y0+ysize,x0:x0+xsize])
    gradcp=np.gradient(demrdrcp,axis=1)
    
    sigma_amp=150
    sigma_dem=0.5
    
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
    
    
    gradfilt=denoise_nl_means(gradcp, h=0.6 * sigma_dem, sigma=sigma_dem,
                                 fast_mode=True, **patch_kw)
    
    ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
    ampscp[np.isnan(ampscp)]=0
    ampsfilt=denoise_nl_means(ampscp, h=0.6 * sigma_amp, sigma=sigma_amp,
                                 fast_mode=True, **patch_kw)
    
    result = minimize(residualsarc,x0=np.array([1.0,0.0]),args=(gradfilt,ampsfilt), method='Nelder-Mead', tol=1e-6)
    a,d=result.x
    
    qgrad=(ampsfilt-d)/a
    
    ampscp1=np.copy(amps1[y0:y0+ysize,x0:x0+xsize])
    ampscp1[np.isnan(ampscp1)]=0
    ampsfilt1=denoise_nl_means(ampscp1, h=0.6 * sigma_amp, sigma=sigma_amp,
                                 fast_mode=True, **patch_kw)
    
    result = minimize(residualsarc,x0=np.array([1.0,0.0]),args=(gradfilt,ampsfilt1), method='Nelder-Mead', tol=1e-6)
    a1,d1=result.x
    
    qgrad1=(ampsfilt1-d1)/a1
    
    dif=qgrad1-qgrad
    
    filt,coh,std=coherence(dif,win=10,thres=1)
    
    #filt[np.abs(filt)>0.6]=0
    if not off==None:
        filt-=off
    else:
        off=np.nanmean(filt)
        filt-=off
    
    #demdif,x,y=integrate(filt,row=row,col=col,std=std)
    filt[filt==-1*off]=0
    filt[np.isnan(filt)]=0
    
    return gradcp,gradfilt,ampscp,ampscp1,ampsfilt,ampsfilt1,qgrad,qgrad1,dif,filt,coh,std


def sartodif2(amps,amps1,grad,los,x0=0,y0=0,xsize=0,ysize=0):
    if xsize==0:
        xsize=amps.shape[1]
    if ysize==0:
        ysize=amps.shape[0]
    gradcp=np.copy(grad[y0:y0+ysize,x0:x0+xsize])
    gradfilt=gaussian_filter(gradcp, sigma=10)
    
    loscp=np.copy(los[y0:y0+ysize,x0:x0+xsize])
    
    ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
    ampscp[np.isnan(ampscp)]=0
    ampsfilt=gaussian_filter(ampscp, sigma=5)
    
    result = minimize(residualsar,x0=np.array([1.0,0.0,0.0]),args=(gradfilt,ampsfilt,loscp), method='Nelder-Mead', tol=1e-6)
    a,b,c=result.x
    
    qgrad=(ampsfilt-b*loscp-c)/a
    
    ampscp1=np.copy(amps1[y0:y0+ysize,x0:x0+xsize])
    ampscp1[np.isnan(ampscp1)]=0
    ampsfilt1=gaussian_filter(ampscp1, sigma=5)
    
    result = minimize(residualsar,x0=np.array([1.0,0.0,0.0]),args=(gradfilt,ampsfilt1,loscp), method='Nelder-Mead', tol=1e-6)
    a1,b1,c1=result.x
    
    qgrad1=(ampsfilt1-b1*loscp-c1)/a1
    
    dif=(ampsfilt1-ampsfilt)
    demdif=np.cumsum((dif-(b1-b)*loscp-(c1-c))/(a1-a),axis=1)
    residualcol=np.array([np.sum(np.abs(demdif[:,i])) for i in range(demdif.shape[1])])
    pos=np.argmin(residualcol)

    for i in range(demdif.shape[0]):
        offset=demdif[i,pos]
        demdif[i,:]+=offset

    for i in range(demdif.shape[1]):
        demdif[:,i]=gaussian_filter1d(demdif[:,i], sigma=30)
    
    return gradcp,gradfilt,loscp,ampscp,ampscp1,ampsfilt,ampsfilt1,qgrad,qgrad1,demdif

def sartodif3(amps,amps1,demrdr,x0=0,y0=0,xsize=0,ysize=0):
    if xsize==0:
        xsize=amps.shape[1]
    if ysize==0:
        ysize=amps.shape[0]
        
    demrdrcp=np.copy(demrdr[y0:y0+ysize,x0:x0+xsize])
    gradx=np.gradient(demrdrcp,axis=1)
    grady=np.gradient(demrdrcp,axis=0)
    
    gradxfilt=gaussian_filter(gradx, sigma=10)
    gradyfilt=gaussian_filter(grady, sigma=10)
    
    sigma=1
    
    ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
    ampscp[np.isnan(ampscp)]=0
    ampsfilt=gaussian_filter(ampscp, sigma=5)
    
    result = minimize(residualsarc1,x0=np.array([1.0,0.0]),args=(gradxfilt,ampsfilt), method='Nelder-Mead', tol=1e-6)
    a,b=result.x
    
    #qgrad=(ampsfilt-b)/(a*sigma)
    qgrad=(ampsfilt-b)/a
    
    ampscp1=np.copy(amps1[y0:y0+ysize,x0:x0+xsize])
    ampscp1[np.isnan(ampscp1)]=0
    ampsfilt1=gaussian_filter(ampscp1, sigma=5)
    
    result = minimize(residualsarc1,x0=np.array([1.0,0.0]),args=(gradxfilt,ampsfilt1), method='Nelder-Mead', tol=1e-6)
    a1,b1=result.x
    
    qgrad1=(ampsfilt1-b1)/(a1*sigma)
    
    dif=qgrad1-qgrad
    demdif=integrate(dif)
    
    return gradx,gradxfilt,grady,gradyfilt,ampscp,ampscp1,ampsfilt,ampsfilt1,qgrad,qgrad1,demdif

def sartodif4(amps,amps1,demrdr,x0=0,y0=0,xsize=0,ysize=0):
    if xsize==0:
        xsize=amps.shape[1]
    if ysize==0:
        ysize=amps.shape[0]
        
    demrdrcp=np.copy(demrdr[y0:y0+ysize,x0:x0+xsize])
    gradx=np.gradient(demrdrcp,axis=1)
    grady=np.gradient(demrdrcp,axis=0)
    
    gradxfilt=gaussian_filter(gradx, sigma=10)
    gradyfilt=gaussian_filter(grady, sigma=10)
    
    sigma=1
    
    ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
    ampscp[np.isnan(ampscp)]=0
    ampsfilt=gaussian_filter(ampscp, sigma=5)
    
    result = minimize(residualsarc1,x0=np.array([1.0,0.0]),args=(gradxfilt,ampsfilt), method='Nelder-Mead', tol=1e-6)
    a,b=result.x
    
    qgrad=(ampsfilt-b)/a
    
    ampscp1=np.copy(amps1[y0:y0+ysize,x0:x0+xsize])
    ampscp1[np.isnan(ampscp1)]=0
    ampsfilt1=gaussian_filter(ampscp1, sigma=5)
    
    result = minimize(residualsarc1,x0=np.array([1.0,0.0]),args=(gradxfilt,ampsfilt1), method='Nelder-Mead', tol=1e-6)
    a1,b1=result.x
    
    qgrad1=(ampsfilt1-b1)/(a1*sigma)
    
    dif=qgrad1-qgrad
    filt,coh=coherence(dif,win=10,thres=1)
    filt[np.isnan(filt)]=0
    demdif,x,y=integrate(filt)
    
    return gradx,gradxfilt,grady,gradyfilt,ampscp,ampscp1,ampsfilt,ampsfilt1,qgrad,qgrad1,demdif,x,y

def coherence(dif,win=5,thres=5):
    filt=np.zeros(dif.shape)
    coh=np.ones(dif.shape)
    std=np.zeros(dif.shape)
    for i in range(dif.shape[1]):
        for j in range(dif.shape[0]):
            if i-win<0:
                coli=0
                cold=i+win+1
            elif i+win>=dif.shape[1]:
                coli=i-win
                cold=dif.shape[1]
            else:
                coli=i-win
                cold=i+win+1
            if j-win<0:
                rowi=0
                rowd=j+win+1
            elif j+win>=dif.shape[0]:
                rowi=j-win
                rowd=dif.shape[0]
            else:
                rowi=j-win
                rowd=j+win+1
            std[j,i]=np.std(dif[rowi:rowd,coli:cold])
            if std[j,i]>thres:
                filt[j,i]=np.nan
                coh[j,i]=0
            else:
                filt[j,i]=dif[j,i]
            
                
    return filt,coh,std

def integrate(dif,row=None,col=None,std=[]):
    print('Dif',dif)
    demdif=np.nancumsum(dif,axis=1)
    print(np.nanmax(dif),np.nanmin(dif))
    x=[]
    y=[]
    if row==None and col==None:
        minoff=10000
        row=0
        for i in range(demdif.shape[0]):
            pos=None
            if len(std)==0:
                pos=np.nanargmin(np.abs(dif[i,:]))
            else:
                sort=np.argsort(np.abs(dif[i,:]))
                for p in sort:
                    if std[i,p]<0.05:
                        pos=p
                        break
                if pos==None:
                    try:
                        pos=np.nanargmin(np.abs(dif[i,:])+std[i,:])
                    except:
                        continue
            offset=dif[i,pos]
            #demdifff[i,:]-=offset
            if abs(offset)<minoff:
                row=i
                col=pos
                minoff=abs(offset)

    updone=False
    downdone=False
    i=row
    j=col
    k=row
    m=col
    x.append(col)
    y.append(row)
    print(demdif[row,:],demdif[row,col],row,col)
    demdif[row,:]-=demdif[row,col]
    while(not (updone and downdone)):
        anj=20
        dej=20
        anm=20
        dem=20
        if j+40>=demdif.shape[1]:
            dej=demdif.shape[1]-1
        if j-40<0:
            anj=j
        if m+40>=demdif.shape[1]:
            dem=demdif.shape[1]-1
        if m-40<0:
            anm=m
        if i+1==demdif.shape[0]:
            downdone=True
        else:
            i=i+1
            j=j-anj+np.nanargmin(np.abs(dif[i,j-anj:j+dej]))
            demdif[i,:]-=demdif[i,j]
            x.append(j)
            y.append(i)
        if k-1==-1:
            updone=True
        else:
            k=k-1
            m=m-anm+np.nanargmin(np.abs(dif[k,m-anm:m+dem]))
            demdif[k,:]-=demdif[k,m]
            x.append(m)
            y.append(k)
            
    return demdif,x,y

def plot_rsl(matriz,fmt='grd',title=''):
    plt.figure()
    if fmt=='grd':
        plt.title('Gradient DEM')
        plt.imshow(matriz,cmap='gray',vmin=-0.5,vmax=0.5)
        clb=plt.colorbar()
        clb.set_label('meters/px')
    elif fmt=='amp':
        plt.title('Amplitude')
        plt.imshow(matriz,cmap='gray',vmin=0,vmax=200)
        plt.colorbar()
    else:
        plt.imshow(matriz,cmap='gray')
        clb=plt.colorbar()
        clb.set_label('meters')
    if not title=='':
        plt.title(title)
    plt.xlabel('Range coordinates (px)')
    plt.ylabel('Azimuth coordinates (px)')
    plt.show()

def modify_xml(xml,ruta):
    archivo=open(xml,'r')
    lineas=archivo.readlines()
    archivo.close()
    subprocess.call('rm -rf '+xml,shell=True)
    archivo=open(xml,'w')
    modify=False
    for linea in lineas:
        if modify==True:
            archivo.write(ruta+'\n')
            fecha=ruta.split('______')[1].split('_')[3].split('T')[0]
            modify=False
        else:
            if '<property name=\"OUTPUT\">' in linea:
                archivo.write('<property name=\"OUTPUT\">'+fecha+'</property>\n')
            else:
                archivo.write(linea)
        if '<property name=\"XML\">' in linea:
            modify=True
    archivo.close()

def insar(file1,file2,name='temporal'):
    print('insar of ',file1,file2)
    if os.path.exists(name):
        subprocess.call('rm -rf '+name,shell=True)
    subprocess.call('cp -r template '+name,shell=True)
    modify_xml(name+'/reference.xml',file1)
    modify_xml(name+'/secondary.xml',file2)
    cwd=os.getcwd()
    os.chdir('./'+name+'/')
    subprocess.call('/home/jovyan/.local/envs/insar_analysis/bin/python /home/jovyan/.local/envs/insar_analysis/lib/python3.8/site-packages/isce/applications/stripmapApp.py ./stripmapApp.xml',shell=True)
    os.chdir(cwd)

def un_time_series(i,archivos):
    basename1=os.path.basename(archivos[i])
    print(basename1)
    fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
    file1='../../data/slcs1/'+basename1+'/'+basename1+'.xml'
    basename2=os.path.basename(archivos[i+1])
    fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
    file2='../../data/slcs1/'+basename2+'/'+basename2+'.xml'
    try:
        name='temporal/'+fecha1+'_'+fecha2
        insar(file1,file2,name=name)
        subprocess.call('mkdir projections/'+fecha1+'_'+fecha2,shell=True)
        subprocess.call('cp -r '+name+'/geometry projections/'+fecha1+'_'+fecha2+'/',shell=True)
        subprocess.call('cp -r '+name+'/*_slc projections/'+fecha1+'_'+fecha2+'/',shell=True)
        subprocess.call('cp -r '+name+'/coregisteredSlc projections/'+fecha1+'_'+fecha2+'/',shell=True)
        subprocess.call('rm -rf '+name,shell=True)
    except:
        return 'd','d'
    return 'd','d'

def time_series_paralell(nproc=4,name='projections'):
    archivos=sorted(glob.glob('./data/slcs1/*'))
    subprocess.call('mkdir temporal',shell=True)
    if not os.path.exists('projections'):
        subprocess.call('mkdir projections',shell=True)
    runs=int((len(archivos)-1)/nproc)
    if not runs*nproc==len(archivos)-1:
        runs+=1
    
    for run in range(runs):
        if run<runs-1:
            archivost=archivos[run*nproc:(run+1)*nproc+1]
        else:
            archivost=archivos[run*nproc::]
        pool = multiprocessing.Pool(processes=nproc)
        subrutina=partial(un_time_series,archivos=archivost)
        tempi,tempu=zip(*pool.map(subrutina, range(len(archivost)-1)))
        pool.close()
        pool.join()
        if os.path.exists('./'+name+'.h5'):
            h5f = h5py.File('./'+name+'.h5', 'a')
        else:
            h5f = h5py.File('./'+name+'.h5', 'w')
        outputs=glob.glob('./projections/*')
        for archivo in outputs:
            basename=os.path.basename(archivo)
            print(basename)
            fecha1=basename.split('_')[0]
            fecha2=basename.split('_')[1]
            grp=h5f.create_group(basename)
            demrdr=readdata('./projections/'+basename+'/z.rdr.full')
            lonrdr=readdata('./projections/'+basename+'/lon.rdr.full')
            latrdr=readdata('./projections/'+basename+'/lat.rdr.full')
            amps,angles=readcomplexdata('./projections/'+basename+'/'+fecha1+'_slc/'+fecha1+'.slc.vrt')
            amps1,angles1=readcomplexdata('./projections/'+basename+'/coregisteredSlc/coarse_coreg.slc.vrt')
            grp.create_dataset('dem', data=demrdr, compression="gzip")
            grp.create_dataset('lon', data=lonrdr, compression="gzip")
            grp.create_dataset('lat', data=latrdr, compression="gzip")
            grp.create_dataset('amps', data=amps, compression="gzip")
            grp.create_dataset('amps1', data=amps1, compression="gzip")
        h5f.close()
        subprocess.call('rm -rf ./projections/*',shell=True)
    #subprocess.call('rm -rf temporal',shell=True)

def time_series(index=0):
    archivos=sorted(glob.glob('./data/slcs/*'))
    for i in range(len(archivos[index:index+1])):
        i+=index
        basename1=os.path.basename(archivos[i])
        fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
        file1='../data/slcs/'+basename1+'/'+basename1+'.xml'
        basename2=os.path.basename(archivos[i+1])
        fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
        file2='../data/slcs/'+basename2+'/'+basename2+'.xml'
        insar(file1,file2)
        subprocess.call('mkdir projections/'+fecha1+'_'+fecha2,shell=True)
        subprocess.call('cp -r temporal/geometry projections/'+fecha1+'_'+fecha2+'/',shell=True)
        subprocess.call('cp -r temporal/*_slc projections/'+fecha1+'_'+fecha2+'/',shell=True)
        subprocess.call('cp -r temporal/coregisteredSlc projections/'+fecha1+'_'+fecha2+'/',shell=True)
        subprocess.call('rm -rf temporal',shell=True)

def time_series2(index=0):
    archivos=sorted(glob.glob('./data/slcs1/*'))
    for i in range(len(archivos[index::])):
        i+=index
        basename1=os.path.basename(archivos[i])
        fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
        file1='../data/slcs1/'+basename1+'/'+basename1+'.xml'
        basename2=os.path.basename(archivos[i+1])
        fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
        file2='../data/slcs1/'+basename2+'/'+basename2+'.xml'
        try:
            insar(file1,file2)
            demrdr=readdata('./temporal/geometry/z.rdr.full')
            lonrdr=readdata('./temporal/geometry/lon.rdr.full')
            latrdr=readdata('./temporal/geometry/lat.rdr.full')
            amps,angles=readcomplexdata('./temporal/'+fecha1+'_slc/'+fecha1+'.slc.vrt')
            amps1,angles1=readcomplexdata('./temporal/coregisteredSlc/coarse_coreg.slc.vrt')
            if not os.path.exists('./projections.h5'):
                h5f = h5py.File('./projections.h5', 'w')
            else:
                h5f = h5py.File('./projections.h5', 'a')
            grp=h5f.create_group(fecha1+'_'+fecha2)
            grp.create_dataset('dem', data=demrdr, compression="gzip")
            grp.create_dataset('lon', data=lonrdr, compression="gzip")
            grp.create_dataset('lat', data=latrdr, compression="gzip")
            grp.create_dataset('amps', data=amps, compression="gzip")
            grp.create_dataset('amps1', data=amps1, compression="gzip")
            subprocess.call('rm -rf temporal',shell=True)
            h5f.close()
        except:
            continue

def time_series3(index=1):
    archivos=sorted(glob.glob('./data/slcs1/*'))
    for i in range(len(archivos[index::])):
        i+=index
        basename1=os.path.basename(archivos[0])
        fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
        file1='../data/slcs1/'+basename1+'/'+basename1+'.xml'
        basename2=os.path.basename(archivos[i])
        fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
        file2='../data/slcs1/'+basename2+'/'+basename2+'.xml'
        try:
            insar(file1,file2)
            demrdr=readdata('./temporal/geometry/z.rdr.full')
            lonrdr=readdata('./temporal/geometry/lon.rdr.full')
            latrdr=readdata('./temporal/geometry/lat.rdr.full')
            amps,angles=readcomplexdata('./temporal/'+fecha1+'_slc/'+fecha1+'.slc.vrt')
            amps1,angles1=readcomplexdata('./temporal/coregisteredSlc/coarse_coreg.slc.vrt')
            if not os.path.exists('./projections.h5'):
                h5f = h5py.File('./projections.h5', 'w')
            else:
                h5f = h5py.File('./projections.h5', 'a')
            grp=h5f.create_group(fecha1+'_'+fecha2)
            grp.create_dataset('dem', data=demrdr, compression="gzip")
            grp.create_dataset('lon', data=lonrdr, compression="gzip")
            grp.create_dataset('lat', data=latrdr, compression="gzip")
            grp.create_dataset('amps', data=amps, compression="gzip")
            grp.create_dataset('amps1', data=amps1, compression="gzip")
            subprocess.call('rm -rf temporal',shell=True)
            h5f.close()
        except:
            continue

def time_series4(index=1):
    archivos=sorted(glob.glob('./data/slcs/*'))
    for i in range(len(archivos[index::])):
        i+=index
        basename1=os.path.basename(archivos[0])
        fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
        file1='../data/slcs/'+basename1+'/'+basename1+'.xml'
        basename2=os.path.basename(archivos[i])
        fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
        file2='../data/slcs/'+basename2+'/'+basename2+'.xml'
        try:
            insar(file1,file2)
            demrdr=readdata('./temporal/geometry/z.rdr.full')
            lonrdr=readdata('./temporal/geometry/lon.rdr.full')
            latrdr=readdata('./temporal/geometry/lat.rdr.full')
            amps,angles=readcomplexdata('./temporal/'+fecha1+'_slc/'+fecha1+'.slc.vrt')
            amps1,angles1=readcomplexdata('./temporal/coregisteredSlc/coarse_coreg.slc.vrt')
            if not os.path.exists('./projectionsA.h5'):
                h5f = h5py.File('./projectionsA.h5', 'w')
            else:
                h5f = h5py.File('./projectionsA.h5', 'a')
            grp=h5f.create_group(fecha1+'_'+fecha2)
            grp.create_dataset('dem', data=demrdr, compression="gzip")
            grp.create_dataset('lon', data=lonrdr, compression="gzip")
            grp.create_dataset('lat', data=latrdr, compression="gzip")
            grp.create_dataset('amps', data=amps, compression="gzip")
            grp.create_dataset('amps1', data=amps1, compression="gzip")
            subprocess.call('rm -rf temporal',shell=True)
            h5f.close()
        except:
            continue

def time_series5(index=1):
    archivost=glob.glob('./data/slcs1/*')
    fechas=np.array([int(archivo.split('SRA_')[1].split('_')[0].split('T')[0]) for archivo in archivost])
    indices=np.argsort(fechas)
    archivos=[archivost[i] for i in indices]
    h5i=h5py.File('projections.h5','r')
    keys=[key for key in h5i.keys()]
    h5i.close()
    for i in range(len(archivos[index::])):
        
        i+=index
        basename1=os.path.basename(archivos[0])
        fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
        file1='../data/slcs1/'+basename1+'/'+basename1+'.xml'
        basename2=os.path.basename(archivos[i])
        fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
        file2='../data/slcs1/'+basename2+'/'+basename2+'.xml'
        if os.path.exists('./projectionsD.h5'):
            h5i = h5py.File('./projectionsD.h5', 'r')
            llaves=[key for key in h5i.keys()]
            h5i.close()
            if fecha1+'_'+fecha2 in llaves:
                print(fecha1+'_'+fecha2,' ya esta')
                continue
        if fecha1+'_'+fecha2 in keys:
            key=fecha1+'_'+fecha2
            print(fecha1+'_'+fecha2,' se va a copiar')
            h5i=h5py.File('projections.h5','r')
            dem=h5i[key+'/dem'][:]
            lon=h5i[key+'/lon'][:]
            lat=h5i[key+'/lat'][:]
            amps=h5i[key+'/amps'][:]
            amps1=h5i[key+'/amps1'][:]
            h5i.close()
            if not os.path.exists('./projectionsD.h5'):
                h5f = h5py.File('./projectionsD.h5', 'w')
            else:
                h5f = h5py.File('./projectionsD.h5', 'a')
            grp=h5f.create_group(key)
            grp.create_dataset('dem',data=dem,compression='gzip')
            grp.create_dataset('lon',data=lon,compression='gzip')
            grp.create_dataset('lat',data=lat,compression='gzip')
            grp.create_dataset('amps',data=amps,compression='gzip')
            grp.create_dataset('amps1',data=amps1,compression='gzip')
            h5f.close()
        else:
            try:
                insar(file1,file2)
                demrdr=readdata('./temporal/geometry/z.rdr.full')
                lonrdr=readdata('./temporal/geometry/lon.rdr.full')
                latrdr=readdata('./temporal/geometry/lat.rdr.full')
                amps,angles=readcomplexdata('./temporal/'+fecha1+'_slc/'+fecha1+'.slc.vrt')
                amps1,angles1=readcomplexdata('./temporal/coregisteredSlc/coarse_coreg.slc.vrt')
                if not os.path.exists('./projectionsD.h5'):
                    h5f = h5py.File('./projectionsD.h5', 'w')
                else:
                    h5f = h5py.File('./projectionsD.h5', 'a')
                grp=h5f.create_group(fecha1+'_'+fecha2)
                grp.create_dataset('dem', data=demrdr, compression="gzip")
                grp.create_dataset('lon', data=lonrdr, compression="gzip")
                grp.create_dataset('lat', data=latrdr, compression="gzip")
                grp.create_dataset('amps', data=amps, compression="gzip")
                grp.create_dataset('amps1', data=amps1, compression="gzip")
                subprocess.call('rm -rf temporal',shell=True)
                h5f.close()
            except:
                continue            
            
def time_series_master(master=0,index=0):
    archivos=sorted(glob.glob('./data/slcs/*'))
    for i in range(len(archivos[index+1::])):
        i+=index+1
        basename1=os.path.basename(archivos[index])
        fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
        file1='../data/slcs/'+basename1+'/'+basename1+'.xml'
        basename2=os.path.basename(archivos[i])
        fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
        file2='../data/slcs/'+basename2+'/'+basename2+'.xml'
        insar(file1,file2)
        subprocess.call('mkdir master/'+fecha1+'_'+fecha2,shell=True)
        if i==index+1:
            subprocess.call('cp -r temporal/geometry master/'+fecha1+'_'+fecha2+'/',shell=True)
            subprocess.call('cp -r temporal/*_slc master/'+fecha1+'_'+fecha2+'/',shell=True)
        subprocess.call('cp -r temporal/coregisteredSlc master/'+fecha1+'_'+fecha2+'/',shell=True)
        subprocess.call('rm -rf temporal',shell=True)        

def ll2rc(lons,lats,lon,lat):
    distances=np.sqrt((lons-lon)**2+(lats-lat)**2)
    minpos=np.argwhere(distances==np.nanmin(distances))
    return minpos  

def get_box(lonrdr,latrdr,lons=[-163.977,-163.967],lats=[54.7535,54.7585]):
    y2,x2=ll2rc(lonrdr,latrdr,lons[0],lats[0])[0]
    y3,x3=ll2rc(lonrdr,latrdr,lons[1],lats[1])[0]
    x0=np.min([x2,x3])
    y0=np.min([y2,y3])
    x1=np.max([x2,x3])
    y1=np.max([y2,y3])
    xsize=x1-x0
    ysize=y1-y0
    return x0,y0,xsize,ysize
        
def dif_series():
    archivos=sorted(glob.glob('./projections/*'))
    for archivo in archivos:
        if archivo.endswith('.zip'):
            continue
        basename=os.path.basename(archivo)
        demrdr=readdata('./projections/'+basename+'/z.rdr.full')
        lonrdr=readdata('./projections/'+basename+'/lon.rdr.full')
        latrdr=readdata('./projections/'+basename+'/lat.rdr.full')
        amps,angles=readcomplexdata('./projections/'+basename+'/'+basename.split('_')[0]+'_slc/'+basename.split('_')[0]+'.slc.vrt')
        amps1,angles1=readcomplexdata('./projections/'+basename+'/coregisteredSlc/coarse_coreg.slc.vrt')
        lons=[-163.977,-163.973]
        lats=[54.754,54.758]
        y2,x2=ll2rc(lonrdr,latrdr,lons[0],lats[0])[0]
        y3,x3=ll2rc(lonrdr,latrdr,lons[1],lats[1])[0]
        x0=np.min([x2,x3])
        y0=np.min([y2,y3])
        x1=np.max([x2,x3])
        y1=np.max([y2,y3])
        xsize=x1-x0
        ysize=y1-y0
        print(x0,x1,y0,y1)
        gradcp,gradfilt,ampscp,ampscp1,ampsfilt,ampsfilt1,qgrad,qgrad1,dif,filt,coh,std,demdif =sartodif12(amps,amps1,demrdr,x0=x0,y0=y0,xsize=xsize,ysize=ysize)
        np.save('./results3/'+basename+'_qgrad.npy',qgrad)
        np.save('./results3/'+basename+'_qgrad1.npy',qgrad1)
        np.save('./results3/'+basename+'_amps.npy',amps[y0:y1,x0:x1])
        np.save('./results3/'+basename+'_amps1.npy',amps1[y0:y1,x0:x1])
        np.save('./results3/'+basename+'_lon.npy',lonrdr[y0:y1,x0:x1])
        np.save('./results3/'+basename+'_lat.npy',latrdr[y0:y1,x0:x1])
        np.save('./results3/'+basename+'_dem.npy',demrdr[y0:y1,x0:x1])
        np.save('./results3/'+basename+'_dif.npy',dif)
        np.save('./results3/'+basename+'_filt.npy',filt)
        np.save('./results3/'+basename+'_demdif.npy',demdif)
        
def dif_series2(name,output='shishaldin',lons=[-163.977,-163.963],lats=[54.7535,54.7585]):
    h5f = h5py.File(name+'.h5','r')
    h5o = h5py.File(output+'.h5','w')
    for key in h5f.keys():
        print(key)
        demrdr=h5f[key+'/dem'][:]
        lonrdr=h5f[key+'/lon'][:]
        latrdr=h5f[key+'/lat'][:]
        amps=h5f[key+'/amps'][:]
        amps1=h5f[key+'/amps1'][:]
        y2,x2=ll2rc(lonrdr,latrdr,lons[0],lats[0])[0]
        y3,x3=ll2rc(lonrdr,latrdr,lons[1],lats[1])[0]
        x0=np.min([x2,x3])
        y0=np.min([y2,y3])
        x1=np.max([x2,x3])
        y1=np.max([y2,y3])
        xsize=x1-x0
        ysize=y1-y0
        
        try:
            gradcp,gradfilt,ampscp,ampscp1,ampsfilt,ampsfilt1,qgrad,qgrad1,dif,filt,coh,std,demdif =sartodif12(amps,amps1,demrdr,x0=x0,y0=y0,xsize=xsize,ysize=ysize)
            grp=h5o.create_group(key)
            grp.create_dataset('qgrad', data=qgrad, compression="gzip")
            grp.create_dataset('qgrad1', data=qgrad1, compression="gzip")
            grp.create_dataset('dif', data=dif, compression="gzip")
            grp.create_dataset('filt', data=filt, compression="gzip")
            grp.create_dataset('coh', data=coh, compression="gzip")
            grp.create_dataset('demdif', data=demdif, compression="gzip")
        except:
            print('Error with'+key)
            continue
    h5o.close()
    h5f.close()

def dif_series3(output='shishaldin',lons=[-163.977,-163.967],lats=[54.7515,54.7585]):
    h5i = h5py.File('projections.h5','r')
    h5o = h5py.File(output+'.h5','w')
    for i,key in enumerate(h5i.keys()):
        dem=h5i[key+'/dem'][:]
        amps=h5i[key+'/amps'][:]
        lonrdr=h5i[key+'/lon'][:]
        latrdr=h5i[key+'/lat'][:]
        x0,y0,xsizet,ysizet=get_box(lonrdr,latrdr)
        if i==0:
            xsize=xsizet
            ysize=ysizet
            h5o.create_dataset('dem', data=dem[y0:y0+ysize,x0:x0+xsize], compression="gzip")
        grad=np.gradient(dem,axis=1)[y0:y0+ysize,x0:x0+xsize]
        ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
        ampsfilt,gradfilt,grad_syn=sar2grad2(amps,dem,x0,y0,xsize,ysize)
        res=np.abs(grad_syn-gradfilt)
        grad_syn[res>=1.0]=np.nan
        demcp=np.copy(dem[y0:y0+ysize,x0:x0+xsize])
        gradcp=np.copy(grad_syn)
        gradcp[np.isnan(gradcp)]=gradfilt[np.isnan(gradcp)]
        demsyn=np.nancumsum(gradcp,axis=1)
        gradfiltcp=np.copy(gradfilt)
        gradfiltcp[np.isnan(demsyn)]=np.nan
        demfilt=np.nancumsum(gradfiltcp,axis=1)
        demdef=np.empty(demsyn.shape)
        demdef[:]=np.nan
        for j in range(demsyn.shape[0]):
            rowsyn=demsyn[j,:]
            rowdem=demfilt[j,:]
            posnan=np.argwhere(np.isnan(rowsyn))
            if len(posnan)>0:
                for i,pos in enumerate(posnan):
                    pos=pos[0]
                    if i==0:
                        b=bestoff(rowsyn[0:pos],rowdem[0:pos])
                        demdef[j,0:pos]=rowsyn[0:pos]+b+demcp[j,0]
                        demfilt[j,0:pos]+=demcp[j,0]
                    elif i<len(posnan)-1:
                        if not posnan[i+1][0]==pos+1:
                            b=bestoff(rowsyn[pos+1:posnan[i+1][0]],rowdem[pos+1:posnan[i+1][0]])
                            demdef[j,pos+1:posnan[i+1][0]]=rowsyn[pos+1:posnan[i+1][0]]+b+(demcp[j,pos]-rowsyn[pos+1])
                            demfilt[j,pos+1:posnan[i+1][0]]+=(demcp[j,pos]-demfilt[j,pos])
                    else:
                        b=bestoff(rowsyn[pos+1::],rowdem[pos+1::])
                        demdef[j,pos+1::]=rowsyn[pos+1::]+b+(demcp[j,pos]-rowsyn[pos+1])
                        demfilt[j,pos+1::]+=(demcp[j,pos]-demfilt[j,pos])
            else:
                b=bestoff(rowsyn,rowdem)
                demdef[j,:]=rowsyn+b+demcp[j,0]
                demfilt[j,:]+=demcp[j,0]
        demfilt[np.isnan(demdef)]=np.nan
        demdef1=np.zeros(demdef.shape)
        for i in range(demdef.shape[1]):
            demdef1[:,i]=gaussian_filter1d(demdef[:,i],sigma=20)
        demdef1[np.isnan(grad_syn)]=np.nan
        h5o.create_dataset(key, data=demdef1, compression="gzip")
    h5i.close()
    h5o.close()
    
def dif_series4(output='shishaldin',lons=[-163.977,-163.955],lats=[54.7535,54.7585]):
    h5i = h5py.File('projections.h5','r')
    h5o = h5py.File(output+'.h5','w')
    for i,key in enumerate(h5i.keys()):
        if i==0:
            dem=h5i[key+'/dem'][:]
        amps=h5i[key+'/amps'][:]
        lonrdr=h5i[key+'/lon'][:]
        latrdr=h5i[key+'/lat'][:]
        x0,y0,xsizet,ysizet=get_box(lonrdr,latrdr)
        if i==0:
            xsize=xsizet
            ysize=ysizet
            h5o.create_dataset('dem', data=dem[y0:y0+ysize,x0:x0+xsize], compression="gzip")
        else:
            dem[y0:y0+ysize,x0:x0+xsize]=np.copy(demdef)   
        grad=np.gradient(dem,axis=1)[y0:y0+ysize,x0:x0+xsize]
        ampscp=np.copy(amps[y0:y0+ysize,x0:x0+xsize])
        ampsfilt,gradfilt,grad_syn=sar2grad2(amps,dem,x0,y0,xsize,ysize)
        res=np.abs(grad_syn-gradfilt)
        grad_syn[res>=1.0]=np.nan
        demcp=np.copy(dem[y0:y0+ysize,x0:x0+xsize])
        gradcp=np.copy(grad_syn)
        gradcp[np.isnan(gradcp)]=gradfilt[np.isnan(gradcp)]
        demsyn=np.nancumsum(gradcp,axis=1)
        gradfiltcp=np.copy(gradfilt)
        gradfiltcp[np.isnan(demsyn)]=np.nan
        demfilt=np.nancumsum(gradfiltcp,axis=1)
        demdef=np.empty(demsyn.shape)
        demdef[:]=np.nan
        for j in range(demsyn.shape[0]):
            rowsyn=demsyn[j,:]
            rowdem=demfilt[j,:]
            posnan=np.argwhere(np.isnan(rowsyn))
            if len(posnan)>0:
                for i,pos in enumerate(posnan):
                    pos=pos[0]
                    if i==0:
                        b=bestoff(rowsyn[0:pos],rowdem[0:pos])
                        demdef[j,0:pos]=rowsyn[0:pos]+b+demcp[j,0]
                        demfilt[j,0:pos]+=demcp[j,0]
                    elif i<len(posnan)-1:
                        if not posnan[i+1][0]==pos+1:
                            b=bestoff(rowsyn[pos+1:posnan[i+1][0]],rowdem[pos+1:posnan[i+1][0]])
                            demdef[j,pos+1:posnan[i+1][0]]=rowsyn[pos+1:posnan[i+1][0]]+b+(demcp[j,pos]-rowsyn[pos+1])
                            demfilt[j,pos+1:posnan[i+1][0]]+=(demcp[j,pos]-demfilt[j,pos])
                    else:
                        b=bestoff(rowsyn[pos+1::],rowdem[pos+1::])
                        demdef[j,pos+1::]=rowsyn[pos+1::]+b+(demcp[j,pos]-rowsyn[pos+1])
                        demfilt[j,pos+1::]+=(demcp[j,pos]-demfilt[j,pos])
            else:
                b=bestoff(rowsyn,rowdem)
                demdef[j,:]=rowsyn+b+demcp[j,0]
                demfilt[j,:]+=demcp[j,0]
        demfilt[np.isnan(demdef)]=np.nan
        demdef1=np.zeros(demdef.shape)
        for i in range(demdef.shape[1]):
            demdef1[:,i]=gaussian_filter1d(demdef[:,i],sigma=20)
        demdef1[np.isnan(grad_syn)]=np.nan
        h5o.create_dataset(key, data=demdef1, compression="gzip")
        
    h5i.close()
    h5o.close()
    
def getamps():
    h5i = h5py.File('projections.h5','r')
    h5o = h5py.File('amps.h5','w')
    for i,key in enumerate(h5i.keys()):
        dem=h5i[key+'/dem'][:]
        amps=h5i[key+'/amps'][:]
        lonrdr=h5i[key+'/lon'][:]
        latrdr=h5i[key+'/lat'][:]
        x0,y0,xsizet,ysizet=get_box(lonrdr,latrdr)
        if i==0:
            xsize=xsizet
            ysize=ysizet
        h5o.create_dataset(key.split('_')[0],data=amps[y0:y0+ysize,x0:x0+xsize])
    h5o.close()
    h5i.close()