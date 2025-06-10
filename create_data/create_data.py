import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pyresample import create_area_def
from satpy import Scene
from satpy.writers import get_enhanced_image
from PIL import Image, ImageOps
import os
import skimage
import numpy as np
import pytz
from datetime import timedelta

def get_extent():
    x0 = -2.4e6
    y0 = -2.112e6
    x1 = 1.696e6
    y1 = 1.984e6
    return [x0, y0, x1, y1]

def get_lcc_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                    central_latitude=38.5,
                                    standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                    semiminor_axis=6371229))
    return lcc_proj

def get_scn(fns, to_load, extent, res=2000, reader='abi_l1b'):
    scn = Scene(reader=reader, filenames=fns)
    scn.load(to_load, generate=False)
    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              projection=get_lcc_proj(),
                              resolution=res,
                              area_extent=extent
                              )
    new_scn = scn.resample(my_area) # resamples datasets and resturns a new scene object
    return new_scn

def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def get_IR(scn,bands):
    C14 = scn[bands[0]].compute().data
    #C14 = normalize(C14)
    C15 = scn[bands[1]].compute().data
    #C15 = normalize(C15)
    C16 = scn[bands[2]].compute().data
    #C16 = normalize(C16)
    IR = np.dstack([C14, C15, C16])
    IR[np.isnan(IR)] = 0
    return IR

def get_temp(temp_scn):
    temp = temp_scn['TEMP'].compute().data
    temp[np.isnan(temp)] = 0
    temp[(temp<233) & (temp>0)]= 3 # high
    temp[(temp>=233) & (temp<=253)] = 2 # mid
    temp[(temp>253)] = 1 # low
    temp = np.array(temp, dtype=np.int8)
    therm_mask = thermometer_encode(temp)
    return therm_mask
    #one_hot_mask = get_one_hot(temp) # from the scene object, extract RGB data for plotting
    #return one_hot_mask

def thermometer_encode(x, num_cats=3):
    # n_cats is 3 because [0, 0, 0] is no cloud (ground)
    # [0, 0, 1] is low cloud [1, 1, 1] is high cloud
    flat = x.flatten()
    encoded = (np.arange(num_cats) < flat[:, None]).astype(int)
    return encoded.reshape(*x.shape, num_slots)

def get_one_hot(cat_encode, n_cats=4):
    k = np.take(np.eye(n_cats), cat_encode, axis=1)
    k = k[1:,:,:]
    k = np.einsum('ijk->jki', k)
    return k

def split_and_save(full_image, full_truth, full_coords, fn_head, img_size=1024):
    yr = fn_head.split('s')[1][0:4]
    dn = fn_head.split('s')[1][4:7]
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
            save_loc = "/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Cloud_Top_Temp/cloud_data/"
            skimage.io.imsave('{}data/{}/{}/{}'.format(save_loc, yr, dn, fn), data)
            skimage.io.imsave('{}truth/{}/{}/{}'.format(save_loc, yr, dn, fn), truth)
            fn_list.append(fn)
    return fn_list

def create_data_truth(sat_fns, yr, dn, fn_head):
    sat_fns = list(set(sat_fns))
    sat_fns.sort()
    bands = ['C14', 'C15', 'C16']
    try:
        extent = get_extent()
    except:
        return fn_head
    try:
        scn = get_scn(sat_fns[:-1], bands, extent) # get satpy scn object
    except Exception as e:
        print(e)
        print('{} did not download, moving on'.format(sat_fns))
        for sat_fn in sat_fns:
            if os.path.exists(sat_fn):
                os.remove(sat_fn)
        return fn_head

    conus_crs = scn['C14'].attrs['area'].to_cartopy_crs() # the crs object will have the area extent for plotting
    mask = ['TEMP']
    temp_scn = get_scn([sat_fns[-1]], mask, extent, reader='abi_l2_nc') # get satpy scn object
    therm_mask = get_temp(temp_scn)
    IR = get_IR(scn, bands) # from the scene object, extract RGB data for plotting

    lon, lat = scn['C14'].attrs['area'].get_lonlats()
    lat_lon = np.dstack([lat, lon])
    tif_fns = split_and_save(IR, therm_mask, lat_lon, fn_head)
#    for sat_fn in sat_fns:
#        os.remove(sat_fn)
    print(tif_fns)

    del scn
    del IR
    return

