import matplotlib.pyplot as plt
import sys
import numpy as np
import skimage
from glob import glob
import os
import time
import pytz
from datetime import datetime, timedelta
import wget
import json
from PIL import Image, ImageOps
import s3fs
from pyorbital import astronomy
from satpy import Scene
from pyresample import create_area_def
import cartopy.crs as ccrs

def get_dt_str(dt):
    hr = dt.hour
    hr = str(hr).zfill(2)
    tt = dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    yr = dt.year
    return hr, dn, yr

# get list of datetimes when the solar zenith angle for far east and west boundaries are less than 50 deg
def get_sun_up_dts(dt,min_sza=70, west_lon=-125, west_lat=40, east_lon=-110, east_lat=40):
    dt = dt.replace(hour=0, minute=0)
    day_dts = [dt+timedelta(hours=t) for t in range(24)]
    szas_west = np.asarray([astronomy.sun_zenith_angle(day_dt, west_lon, west_lat) for day_dt in day_dts])
    szas_east = np.asarray([astronomy.sun_zenith_angle(day_dt, east_lon, east_lat) for day_dt in day_dts])
    idx_east = list(np.where(szas_east<min_sza)[0])
    idx_west = list(np.where(szas_west<min_sza)[0])
    overlap = list(set(list(idx_east)) & set(list(idx_west)))
    sun_up_dts = [day_dts[idx] for idx in overlap]
    return sun_up_dts

def get_first_closest_file(band, fns, dt, sat_num):
    diff = timedelta(days=100)
    matching_band_fns = [s for s in fns if band in s]
    for fn in matching_band_fns:
        s_e = fn.split('_')[3:5]
        start = s_e[0]
        s_dt = datetime.strptime(start[1:-3], '%Y%j%H%M')
        s_dt = pytz.utc.localize(s_dt)
        if diff > abs(s_dt - dt):
            diff = abs(s_dt - dt)
            best_start = start
            best_end = s_e[1]
            best_fn = fn
    #fn_str = 'C{}_G{}_{}_{}'.format(band, sat_num, best_start, best_end)
    fn_str = 'G{}_{}_{}'.format(sat_num, best_start, best_end[:-3])
    return best_fn, fn_str

def get_additional_band_file(band, fn_str, fns):
    best_band_fn = [s for s in fns if band in s and fn_str in s]
    return best_band_fn[0]

def get_closest_file(fns, dt, sat_num, bands):
    use_fns = []
    band_init = 'C'+str(bands[0]).zfill(2)
    best_band_fn, fn_str = get_first_closest_file(band_init, fns, dt, sat_num)
    use_fns.append(best_band_fn)
    for band in bands:
        band = 'C'+str(band).zfill(2)
        best_band_fn = get_additional_band_file(band, fn_str, fns)
        use_fns.append(best_band_fn)
    return use_fns

def get_filelist(dt, fs, sat_num, product, scope, bands):
    hr, dn, yr = get_dt_str(dt)
    full_filelist = fs.ls("noaa-goes{}/{}{}/{}/{}/{}/".format(sat_num, product, scope, yr, dn, hr))
    if sat_num == '17' and len(full_filelist) == 0:
        if yr <= 2018:
            sat_num = '16'
            print("YOU WANTED 17 BUT ITS NOT LAUNCHED")
        elif yr >= 2022:
            sat_num = '18'
        full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, scope, yr, dn, hr))
    use_fns = get_closest_file(full_filelist, dt, sat_num, bands)
    return use_fns

def get_first_closest_file_mask(fns, dt, sat_num):
    diff = timedelta(days=100)
    for fn in fns:
        s_e = fn.split('_')[3:5]
        start = s_e[0]
        s_dt = datetime.strptime(start[1:-3], '%Y%j%H%M')
        s_dt = pytz.utc.localize(s_dt)
        if diff > abs(s_dt - dt):
            diff = abs(s_dt - dt)
            best_start = start
            best_end = s_e[1]
            best_fn = fn
    #fn_str = 'C{}_G{}_{}_{}'.format(band, sat_num, best_start, best_end)
    fn_str = 'G{}_{}_{}'.format(sat_num, best_start, best_end[:-3])
    return best_fn

def get_filelist_mask(dt, fs, sat_num, product="ABI-L2-ACHT", scope="F"):
    hr, dn, yr = get_dt_str(dt)
    full_filelist = fs.ls("noaa-goes{}/{}{}/{}/{}/{}/".format(sat_num, product, scope, yr, dn, hr))
    use_fns = get_first_closest_file_mask(full_filelist, dt, sat_num)
    return use_fns

def download_goes(dt, sat_num='18', product='ABI-L1b-Rad', scope='F',bands=list(range(1,4))):
    goes_dir = 'cloud_data/goes/'
    fs = s3fs.S3FileSystem(anon=True)
    use_fns = get_filelist(dt, fs, sat_num, product, scope, bands)
    use_fns_mask = get_filelist_mask(dt, fs, sat_num)
    use_fns.append(use_fns_mask)
    file_locs = []
    for file_path in use_fns:
        fn = file_path.split('/')[-1]
        dl_loc = goes_dir+fn
        file_locs.append(dl_loc)
        if os.path.exists(dl_loc):
            print("{} already exists".format(fn))
        else:
            print('downloading {}'.format(fn))
            fs.get(file_path, dl_loc)
    if len(file_locs) > 0:
        return file_locs
    else:
        print('ERROR NO FILES FOUND FOR TIME REQUESTED: ', dt)

def get_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                     central_latitude=38.5,
                                     standard_parallels=(38.5, 38.5),
                                     globe=ccrs.Globe(semimajor_axis=6371229,
                                                      semiminor_axis=6371229))
    return lcc_proj

# get the Satpy Scene object
def get_scn(fns, to_load, extent, res=3000, proj=get_proj(), reader='abi_l1b', print_info=False):
    scn = Scene(reader=reader, filenames=fns)
    scn.load(to_load, generate=False)
    my_area = create_area_def(area_id='my_area',
                              projection=proj,
                              resolution=res,
                              area_extent=extent
                              )
    if print_info:
        print("Available channels in the Scene object:\n", scn.available_dataset_names())
        print("\nAvailable composites:\n", scn.available_composite_names())
        print("\nArea definitition:\n",my_area)
    new_scn = scn.resample(my_area) # resamples datasets and resturns a new scene object
    return new_scn

def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def get_IR(scn,bands):
    C14 = scn[bands[0]].compute().data
    C15 = scn[bands[1]].compute().data
    C16 = scn[bands[2]].compute().data
    IR = np.dstack([C14, C15, C16])
    IR[np.isnan(IR)] = 0
    IR = normalize(IR)
    return IR

def get_one_hot(binary, n_cats=4):
    k = np.take(np.eye(n_cats), binary, axis=1)
    k = k[1:,:,:]
    k = np.einsum('ijk->jki', k)
    return k

def split_and_save(full_image, full_truth, full_coords, fn_head, img_size=1024):
    yr = fn_head.split('s')[1][0:4]
    n_row = int(full_image.shape[0]/img_size)
    n_col = int(full_image.shape[1]/img_size)
    full_image = full_image[0:int(n_row*img_size),0:int(n_col*img_size)][:]
    full_truth = full_truth[0:int(n_row*img_size),0:int(n_col*img_size)][:]
    full_coords = full_coords[0:int(n_row*img_size),0:int(n_col*img_size)][:]
    fn_list = []
    for row in range(n_row):
        for col in range(n_col):
            data = full_image[int(row*img_size):int((row+1)*img_size),int(col*img_size):int((col+1)*img_size)][:]
            truth = full_truth[int(row*img_size):int((row+1)*img_size),int(col*img_size):int((col+1)*img_size)][:]
            coords = full_coords[int(row*img_size):int((row+1)*img_size),int(col*img_size):int((col+1)*img_size)][:]
            center_lat = np.round(coords[int(img_size/2)][int(img_size/2)][0], 2)
            center_lon = np.round(coords[int(img_size/2)][int(img_size/2)][1], 2)
            fn = '{}_{}_{}.tif'.format(fn_head, center_lat, center_lon)
            save_loc = "./cloud_data/"
            skimage.io.imsave('{}data/{}/{}'.format(save_loc, yr, fn), data)
            skimage.io.imsave('{}truth/{}/{}'.format(save_loc, yr, fn), truth)
            fn_list.append(fn)
    return fn_list

years = ['2023', '2024']
dns = np.linspace(1,365, 365)
for yr in years:
    for dn in dns:
        dt_str = '{}{}'.format(yr, str(int(dn)).zfill(3))
        print(dt_str)
        print(datetime.strptime(dt_str, '%Y%j'))
        day_dt = pytz.utc.localize(datetime.strptime(dt_str, '%Y%j')) # convert to datetime object
        dts = get_sun_up_dts(day_dt)
        print(dts)
        for dt in dts:
            sat_fns = download_goes(dt, sat_num='18', bands=list(range(14,17)))
            fn_head = sat_fns[0].split('C14_')[-1].split('.')[0].split('_e')[0]
            x0 = -2.4e6
            y0 = -2.112e6
            x1 = 1.696e6
            y1 = 1.984e6
            extent = [x0, y0, x1, y1]
            bands = ['C14', 'C15', 'C16']
            res = 2000 # 2km resolution
            scn = get_scn(sat_fns[:-1], bands, extent, res, print_info=True) # get satpy scn object
            conus_crs = scn['C14'].attrs['area'].to_cartopy_crs() # the crs object will have the area extent for plotting
            mask = ['TEMP']
            temp_scn = get_scn([sat_fns[-1]], mask, extent, res, print_info=True, reader='abi_l2_nc') # get satpy scn object

            temp = temp_scn['TEMP'].compute().data
            temp[np.isnan(temp)] = 0
            temp[(temp<233) & (temp>0)]= 3 # high
            temp[(temp>=233) & (temp<=253)] = 2 # mid
            temp[(temp>253)] = 1 # low
            temp = np.array(temp, dtype=np.int8)
            one_hot_mask = get_one_hot(temp) # from the scene object, extract RGB data for plotting

            IR = get_IR(scn, bands) # from the scene object, extract RGB data for plotting

            lon, lat = scn['C14'].attrs['area'].get_lonlats()
            lat_lon = np.dstack([lat, lon])
            tif_fns = split_and_save(IR, one_hot_mask, lat_lon, fn_head)
            x = input('stop!')
