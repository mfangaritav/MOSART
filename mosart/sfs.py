import glob
import logging
import multiprocessing
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from shutil import copyfile, move

import asf_search
import h5py
import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma

gdal.UseExceptions()

log = logging.getLogger(__name__)


def coreg_alos(reference_scene: str, secondary_scene: str, roi: list =None, download: bool =True) -> Path:
    """Create a Stripmap interferogram

    Args:
        reference_scene: Reference scene name
        secondary_scene: Secondary scene name
        download: If True it will download a dem, if False it will look for the file dem/full_res.dem.wgs84
    """
    import zipfile
    from hyp3_isce2 import stripmapapp_alos as stripmapapp
    from hyp3_isce2.dem import download_dem_for_isce2
    from hyp3_isce2.logger import configure_root_logger
    from shapely.geometry.polygon import Polygon
    
    configure_root_logger()
    cwd = os.getcwd()
    process_dir = Path('coreg')
    process_dir.mkdir(exist_ok=True, parents=True)
    os.chdir(process_dir.name)
    
    scenes = sorted([reference_scene, secondary_scene])
    reference_scene = scenes[0]
    secondary_scene = scenes[1]
    products = asf_search.search(
        granule_list=[reference_scene, secondary_scene],
        processingLevel="L1.0",
    )
    if products[0].properties['sceneName']==reference_scene:
        reference_product = products[0]
        secondary_product = products[1]
    else:
        reference_product = products[1]
        secondary_product = products[0]
    assert reference_product.properties['sceneName'] == reference_scene
    assert secondary_product.properties['sceneName'] == secondary_scene
    products = (reference_product, secondary_product)
    polygons = [Polygon(product.geometry['coordinates'][0]) for product in products]
    insar_roi = polygons[0].intersection(polygons[1]).bounds
    print(insar_roi)
    if download:
        dem_path = download_dem_for_isce2(insar_roi, dem_name='glo_30', dem_dir=Path('dem'), buffer=0)
        urls = [product.properties['url'] for product in products]
        zip_paths = [product.properties['fileName'] for product in products]
    else:
        dem_path = Path('dem/full_res.dem.wgs84')
        urls = [product.properties['url'] for product in products[1::]]
        zip_paths = [product.properties['fileName'] for product in products[1::]]

    asf_search.download_urls(urls=urls, path=os.getcwd(), processes=2)

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall()
        os.remove(zip_path)

    reference_image = get_product_file(reference_product, 'IMG-')
    reference_leader = get_product_file(reference_product, 'LED-')

    secondary_image = get_product_file(secondary_product, 'IMG-')
    secondary_leader = get_product_file(secondary_product, 'LED-')

    if roi is None:
        roi=insar_roi
    else:
        roi=(roi[0],roi[1],roi[2],roi[3])

    config = stripmapapp.StripmapappConfig(
        reference_image=reference_image,
        reference_leader=reference_leader,
        secondary_image=secondary_image,
        secondary_leader=secondary_leader,
        roi=roi,
        dem_filename=str(dem_path),
        azimuth_looks=1,
        range_looks=1,
    )
    config_path = config.write_template('stripmapApp.xml')
    with open('stripmapApp.xml', 'r') as file:
        content = file.read()
    pre="<property name=\"do rubbersheetingAzimuth\">True</property>"
    post="<property name=\"reference doppler method\">useDOPIQ</property>"
    modified_content = content.replace(pre,post)

    with open('stripmapApp.xml', 'w') as file:
        file.write(modified_content)

    with open('stripmapApp.xml', 'r') as file:
        content = file.read()

    pre="<property name=\"do rubbersheetingRange\">True</property>"
    post="<property name=\"secondary doppler method\">useDOPIQ</property>"
    modified_content = content.replace(pre, post)

    with open('stripmapApp.xml', 'w') as file:
        file.write(modified_content)
        
    try:
       multiprocessing.set_start_method(None, force=True)
    except RuntimeError:
       pass
    stripmapapp.run_stripmapapp(start='startup', end='refined_resample', config_xml=config_path)
    os.chdir(cwd)


def coreg_tops_burst(
    reference_scene: str,
    secondary_scene: str,
    azimuth_looks: int = 1,
    range_looks: int = 1,
    apply_water_mask: bool = False,
    download: bool = True,
) -> Path:
    """Create a burst interferogram

    Args:
        reference_scene: Reference burst name
        secondary_scene: Secondary burst name
        swath_number: Number of swath to grab bursts from (1, 2, or 3) for IW
        azimuth_looks: Number of azimuth looks
        range_looks: Number of range looks
        apply_water_mask: Whether to apply a pre-unwrap water mask
        download: If True it will download a dem, if False it will look for the file dem/full_res.dem.wgs84
    """
    from hyp3_isce2 import topsapp
    from hyp3_isce2.burst import download_bursts, get_burst_params, get_isce2_burst_bbox, get_region_of_interest
    from hyp3_isce2.dem import download_dem_for_isce2
    from hyp3_isce2.logger import configure_root_logger
    from hyp3_isce2.s1_auxcal import download_aux_cal
    from hyp3_isce2.utils import image_math, isce2_copy, resample_to_radar_io
    from hyp3_isce2.water_mask import create_water_mask
    from isceobj.TopsProc.runMergeBursts import multilook
    from s1_orbits import fetch_for_scene
    
    configure_root_logger()
    cwd = os.getcwd()
    process_dir = Path('coreg')
    process_dir.mkdir(exist_ok=True, parents=True)
    os.chdir(process_dir.name)
    
    orbit_dir = Path('orbits')
    aux_cal_dir = Path('aux_cal')
    dem_dir = Path('dem')
    
    swath_number = int(reference_scene.split('_')[2][2])

    ref_params = get_burst_params(reference_scene)
    sec_params = get_burst_params(secondary_scene)

    if download:
        ref_metadata, sec_metadata = download_bursts([ref_params, sec_params])
        is_ascending = ref_metadata.orbit_direction == 'ascending'
    else:
        sec_metadata = download_bursts([sec_params])
        is_ascending = sec_metadata[0].orbit_direction == 'ascending'

    ref_footprint = get_isce2_burst_bbox(ref_params)
    sec_footprint = get_isce2_burst_bbox(sec_params)

    insar_roi = get_region_of_interest(ref_footprint, sec_footprint, is_ascending=is_ascending)
    dem_roi = ref_footprint.intersection(sec_footprint).bounds

    if abs(dem_roi[0] - dem_roi[2]) > 180.0 and dem_roi[0] * dem_roi[2] < 0.0:
        raise ValueError('Products that cross the anti-meridian are not currently supported.')

    log.info(f'InSAR ROI: {insar_roi}')
    log.info(f'DEM ROI: {dem_roi}')

    if download:
        dem_path = download_dem_for_isce2(dem_roi, dem_name='glo_30', dem_dir=dem_dir, buffer=0, resample_20m=False)
        granules = (ref_params.granule, sec_params.granule)
    else:
        granules = (sec_params.granule)
        dem_path = Path('./dem/full_res.dem.wgs84')
    download_aux_cal(aux_cal_dir)

    if range_looks == 5:
        geocode_dem_path = download_dem_for_isce2(
            dem_roi, dem_name='glo_30', dem_dir=dem_dir, buffer=0, resample_20m=True
        )
    else:
        geocode_dem_path = dem_path

    orbit_dir.mkdir(exist_ok=True, parents=True)
    if download:
        for granule in granules:
            log.info(f'Downloading orbit file for {granule}')
            orbit_file = fetch_for_scene(granule, dir=orbit_dir)
            log.info(f'Got orbit file {orbit_file} from s1_orbits')
    else:
        log.info(f'Downloading orbit file for {granules}')
        orbit_file = fetch_for_scene(granules, dir=orbit_dir)
        log.info(f'Got orbit file {orbit_file} from s1_orbits')

    config = topsapp.TopsappBurstConfig(
        reference_safe=f'{ref_params.granule}.SAFE',
        secondary_safe=f'{sec_params.granule}.SAFE',
        polarization=ref_params.polarization,
        orbit_directory=str(orbit_dir),
        aux_cal_directory=str(aux_cal_dir),
        roi=insar_roi,
        dem_filename=str(dem_path),
        geocode_dem_filename=str(geocode_dem_path),
        swaths=swath_number,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
    )
    config_path = config.write_template('topsApp.xml')

    topsapp.run_topsapp_burst(start='startup', end='preprocess', config_xml=config_path)
    topsapp.swap_burst_vrts()
    if apply_water_mask:
        topsapp.run_topsapp_burst(start='computeBaselines', end='fineresamp', config_xml=config_path)
        water_mask_path = 'water_mask.wgs84'
        create_water_mask(str(dem_path), water_mask_path)
        multilook('merged/lon.rdr.full', outname='merged/lon.rdr', alks=azimuth_looks, rlks=range_looks)
        multilook('merged/lat.rdr.full', outname='merged/lat.rdr', alks=azimuth_looks, rlks=range_looks)
        resample_to_radar_io(water_mask_path, 'merged/lat.rdr', 'merged/lon.rdr', 'merged/water_mask.rdr')
        isce2_copy('merged/phsig.cor', 'merged/unmasked.phsig.cor')
        image_math('merged/unmasked.phsig.cor', 'merged/water_mask.rdr', 'merged/phsig.cor', 'a*b')
        isce2_copy('merged/unmasked.phsig.cor', 'merged/phsig.cor')
    else:
        topsapp.run_topsapp_burst(start='computeBaselines', end='fineresamp', config_xml=config_path)
    os.chdir(cwd)


def get_product_file(product: asf_search.ASFProduct, file_prefix: str) -> str:
    """Path of downloaded file

    Args:
        product: Result from the asf_search query
        file_prefix: Prefix of file of interest (e.g., IMG-, LED-)

    Returns:
        Path of downloaded file
    """
    paths = glob.glob(str(Path(product.properties['fileID']) / f'{file_prefix}*'))
    print(product.properties['fileID'])
    assert len(paths) > 0
    return paths[0]


def preprocessing(lons=None, lats=None, xs=None, ys=None, projections='projections.h5', output='preprocessed.h5',sigma_amp=30, patch_kw=None, gaussian=False):
    """Preprocessing of amplitude images to crop the amplitude, DEM and longitude and latitude files according to the bounding box

    Args:
        lons: longitudes of bounding box in the format [minlon, maxlon]
        lats: latitudes of bounding box in the format [minlat, maxlat]
        xs: range coordinates of the bounding box in the format [minrange, maxrange] (note: if this is defined there's no need to define lons and lats)
        ys: azimuth coordinates of the bounding box in the format [minazimuth, maxazimuth] (note: if this is defined there's no need to define lons and lats)
        projections: name of h5file that will contain the coregistered dem, amplitude images and longitude and latitude files.
        output: file name for the output h5 file
        sigma_amp: non-local means filter strength on the amplitude image
        patch_kw: dictionary with the keys 'patch_size' and 'patch_distance', which are the spatial filter strength in pixels
        gaussian: If True a gaussian filter will be applied after the non-local means filter
    """
    if not (lons is None or lats is None) and not (xs is None or ys is None):
        raise Exception('You have geographic and radar coordinates please choose one')
    
    sigma_dem=0.5
    if patch_kw is None:
        patch_kw = dict(patch_size=5,      # 5x5 patches
                        patch_distance=5,  # 13x13 search area
                        )
    h5i = h5py.File(projections,'r')
    llaves=[key for key in h5i if key.isdigit()]
    h5i.close()
    for i,key in enumerate(llaves):
        h5i = h5py.File(projections,'r')
        if i==0:
            lonrdr=h5i['lon'][:]
            latrdr=h5i['lat'][:]
            dem=h5i['dem'][:]
        amps=h5i[key][:]
        h5i.close()

        if not lons is None or not lats is None:
            x0,y0,xsizet,ysizet=get_box(lonrdr,latrdr,lons=lons,lats=lats)
        else:
            x0=xs[0]
            y0=ys[0]
            xsizet=xs[-1]-xs[0]
            ysizet=ys[-1]-ys[0]

        ampscp=np.copy(amps[y0:y0+ysizet,x0:x0+xsizet])

        if np.sum(np.isnan(ampscp))>0:
            mask1 = np.isnan(ampscp)
            ampscp[mask1] = np.interp(np.flatnonzero(mask1), np.flatnonzero(~mask1), ampscp[~mask1])

        print('Preprocessing for ',key)
        if i==0:
            grd=denoise_nl_means(np.gradient(dem,axis=1)[y0:y0+ysizet,x0:x0+xsizet], h=0.6 * sigma_dem, sigma=sigma_dem, fast_mode=True, **patch_kw)
        ampscp=denoise_nl_means(ampscp, h=0.6 * sigma_amp, sigma=sigma_amp, fast_mode=True, **patch_kw)
        if gaussian:
            ampscp=gaussian_filter(ampscp, sigma=1)

        demcp=dem[y0:y0+ysizet,x0:x0+xsizet]

        loncp=lonrdr[y0:y0+ysizet,x0:x0+xsizet]
        latcp=latrdr[y0:y0+ysizet,x0:x0+xsizet]

        if i==0 and os.path.exists(output):
            subprocess.call('rm -rf '+output, shell=True)

        if os.path.exists(output):
            h5o=h5py.File(output,'a')
        else:
            h5o=h5py.File(output,'w')
            h5o.create_dataset('grad', data=grd, compression="gzip")
            h5o.create_dataset('dem', data=demcp, compression="gzip")
            h5o.create_dataset('lon', data=loncp, compression="gzip")
            h5o.create_dataset('lat', data=latcp, compression="gzip")
        h5o.create_dataset(key, data=ampscp, compression="gzip")
        h5o.close()


def coregistration_alos(alos_list, output='projections_alos.h5',roi=None,index=1):
    """Coregistration of ALOS images

    Args:
        alos_list: text file with the dates and scene names (e.g., '20070705,ALPSRP076921060') 
        output: name of the h5file that will contain the coregistered dem, amplitude images and longitude and latitude files
        index: initial index of the reference scene (default=1) useful to add new coregistered amplitude images
    """
    lista = open(alos_list, 'r')
    lineas = lista.readlines()
    lista.close()
    fechas = sorted(list(set([linea.split(',')[0] for linea in lineas])))
    lineas = sorted(list(set([linea.split(',')[1][0:15] for linea in lineas])))

    for i,linea in enumerate(lineas[index::]):
        reference = lineas[0]
        secondary = linea
        fecha_ref=fechas[0]
        fecha_sec=fechas[index+i]
        if i==0:
            coreg_alos(reference, secondary, roi=roi)
            geom_folder = './coreg/geometry'
            demrdr=readdata(glob.glob(geom_folder+'/z*.vrt')[0])
            lonrdr=readdata(glob.glob(geom_folder+'/lon*.vrt')[0])
            latrdr=readdata(glob.glob(geom_folder+'/lat*.vrt')[0])
            folder_ref = './coreg/reference_slc_crop'
            amps,angles=readcomplexdata(folder_ref+'/reference.slc.vrt')
            folder_sec = './coreg/coregisteredSlc'
            amps1,angles1=readcomplexdata(folder_sec+'/refined_coreg.slc.vrt')
            h5f = h5py.File('./'+output, 'w')
            h5f.create_dataset('dem', data=demrdr, compression="gzip")
            h5f.create_dataset('lon', data=lonrdr, compression="gzip")
            h5f.create_dataset('lat', data=latrdr, compression="gzip")
            h5f.create_dataset(fecha_ref, data=amps, compression="gzip")
            h5f.create_dataset(fecha_sec, data=amps1, compression="gzip")
            h5f.close()
        else:
            coreg_alos(reference, secondary, roi=roi, download=False)
            folder_sec = './coreg/coregisteredSlc'
            amps,angles=readcomplexdata(folder_sec+'/refined_coreg.slc.vrt')
            h5f = h5py.File('./'+output, 'a')
            h5f.create_dataset(fecha_sec, data=amps, compression="gzip")
            h5f.close()
        cwd=os.getcwd()
        os.chdir('coreg')
        subprocess.call('rm -rf SplitSpectrum coregisteredSlc geometry misreg offsets PICKLE reference* secondary* stripmap*', shell=True)
        os.chdir(cwd)
    subprocess.call('rm -rf coreg', shell=True)


def coregistration_bursts(bursts_list, output='projections.h5',index=1):
    """Coregistration of Sentinel-1 bursts

    Args:
        bursts_list: text file with burst names (e.g., 'S1_202224_IW2_20160721T045637_VV_17A3-BURST') 
        output: name of the h5file that will contain the coregistered dem, amplitude images and longitude and latitude files
        index: initial index of the reference scene (default=1) useful to add new coregistered amplitude images
    """
    lista = open(bursts_list, 'r')
    lineas = lista.readlines()
    lista.close()
    lineas = sorted(list(set([linea[0:-1] for linea in lineas])))
    polarizations = list(set([linea.split('_')[4] for linea in lineas]))
    ids = list(set([linea[0:13] for linea in lineas]))

    if len(polarizations)>1:
        raise Exception('The polarizations are not the same in the list')
    elif len(ids)>1:
        raise Exception('There are more than 1 burst id in the list')

    for i,linea in enumerate(lineas[index::]):
        reference = lineas[0]
        secondary = linea
        fecha_ref = reference.split('_')[3].split('T')[0]
        fecha_sec = secondary.split('_')[3].split('T')[0]
        if i==0:
            coreg_tops_burst(reference, secondary)
            geom_folder = glob.glob('./coreg/geom_reference/*')[0]
            demrdr=readdata(glob.glob(geom_folder+'/hgt*.rdr.vrt')[0])
            lonrdr=readdata(glob.glob(geom_folder+'/lon*.rdr.vrt')[0])
            latrdr=readdata(glob.glob(geom_folder+'/lat*.rdr.vrt')[0])
            folder_ref = sorted(glob.glob('./coreg/reference/*'))[0]
            amps,angles=readcomplexdata(glob.glob(folder_ref+'/burst*.slc.vrt')[0])
            folder_sec = sorted(glob.glob('./coreg/fine_coreg/*'))[0]
            amps1,angles1=readcomplexdata(glob.glob(folder_sec+'/burst*.slc.vrt')[0])
            h5f = h5py.File('./'+output, 'w')
            h5f.create_dataset('dem', data=demrdr, compression="gzip")
            h5f.create_dataset('lon', data=lonrdr, compression="gzip")
            h5f.create_dataset('lat', data=latrdr, compression="gzip")
            h5f.create_dataset(fecha_ref, data=amps, compression="gzip")
            h5f.create_dataset(fecha_sec, data=amps1, compression="gzip")
            h5f.close()
        else:
            coreg_tops_burst(reference, secondary, download=False)
            folder_sec = sorted(glob.glob('./coreg/fine_coreg/*'))[0]
            amps,angles=readcomplexdata(glob.glob(folder_sec+'/burst*.slc.vrt')[0])
            h5f = h5py.File('./'+output, 'a')
            h5f.create_dataset(fecha_sec, data=amps, compression="gzip")
            h5f.close()
        cwd=os.getcwd()
        os.chdir('coreg')
        subprocess.call('rm -rf aux_cal fine_* tops* *'+fecha_sec+'* PICKLE secondary', shell=True)
        os.chdir(cwd)
    subprocess.call('rm -rf coreg', shell=True)


def coregistration(output='projections.h5',path=None,index=1):
    """Coregistration of TerraSAR-X images

    Args:
        output: name of the h5file that will contain the coregistered dem, amplitude images and longitude and latitude files
        path: folder path with the SLC products
        index: initial index of the reference scene (default=1) useful to add new coregistered amplitude images
    """
    if path is None:
        archivos=sorted(glob.glob('./data/slcs1/*'))
    else:
        archivos=sorted(glob.glob(path+'/*'))
    for i in range(len(archivos[index::])):
        i+=index
        basename1=os.path.basename(archivos[0])
        fecha1=basename1.split('______')[1].split('_')[3].split('T')[0]
        if path is None:
            prefix='../data/slcs1/'
        else:
            prefix=path+'/'
        file1=prefix+basename1+'/'+basename1+'.xml'
        basename2=os.path.basename(archivos[i])
        fecha2=basename2.split('______')[1].split('_')[3].split('T')[0]
        file2=prefix+basename2+'/'+basename2+'.xml'
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


def insar(file1,file2,name='temporal'):
    print('insar of ',file1,file2)
    isce_path=None
    for path in sys.path:
        if 'site-packages' in path and 'envs' in path:
            isce_path=path+'isce/applications/'
            break
    if not os.path.exists(isce_path):
        raise Exception('ISCE distribution has not been found!!')
    if os.path.exists(name):
        subprocess.call('rm -rf '+name,shell=True)
    subprocess.call('cp -r template '+name,shell=True)
    modify_xml(name+'/reference.xml',file1)
    modify_xml(name+'/secondary.xml',file2)
    cwd=os.getcwd()
    os.chdir('./'+name+'/')
    subprocess.call(sys.executable+' '+isce_path+'/stripmapApp.py ./stripmapApp.xml',shell=True)
    os.chdir(cwd)


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

def georeference(pre_stack,array,pixel_size=0.0001):
    h5i=h5py.File(pre_stack,'r')
    keys=[key for key in h5i.keys() if key.isdigit()]
    lons=h5i['lon'][:]
    lats=h5i['lat'][:]
    h5i.close()
    
    minlon=np.nanmin(lons[~np.isnan(array[0,:,:])])
    maxlon=np.nanmax(lons[~np.isnan(array[0,:,:])])
    minlat=np.nanmin(lats[~np.isnan(array[0,:,:])])
    maxlat=np.nanmax(lats[~np.isnan(array[0,:,:])])

    newlons=np.arange(minlon-pixel_size,maxlon+pixel_size,pixel_size)
    newlats=np.arange(minlat-pixel_size,maxlat+pixel_size,pixel_size)

    geodef=np.empty((array.shape[0],len(newlats),len(newlons)))
    geodef[:,:,:]=np.nan

    for k,change in enumerate(array):
        for i in range(1,len(newlats)-1):
            newlat=newlats[i-1]
            newlatp=newlats[i+1]
            for j in range(1,len(newlons)-1):
                newlon=newlons[j-1]
                newlonp=newlons[j+1]
                cond1=np.logical_and(lons>=newlon,lons<=newlonp)
                cond2=np.logical_and(lats>=newlat,lats<=newlatp)
                cond=np.logical_and(cond1,cond2)
                geodef[k,i,j]=np.nanmean(change[cond])
    
    extent=[minlon,maxlon,minlat,maxlat]
    return geodef[:,::-1,:],extent


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
