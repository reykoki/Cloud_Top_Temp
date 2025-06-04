import shutil
from pyorbital import astronomy
from multiprocessing import Pool
import pickle
import cartopy.crs as ccrs
import glob
import pyproj
import sys
import logging
import geopandas
import os
import random
from datetime import datetime
import numpy as np
import time
import pytz
import cartopy.crs as ccrs
from pyresample import create_area_def
from satpy import Scene
from PIL import Image, ImageOps
import os
import skimage
import shutil
import wget
from datetime import timedelta
from get_goes import get_sat_files, get_goes_dl_loc, get_file_locations, download_sat_files, check_goes_exists

global full_data_dir
full_data_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/cloud_data/'
global goes_dir
goes_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/GOES/'

# get list of datetimes (24 dts per day) chosen random selection per hour
def get_day_dts(dt):
    dt = dt.replace(hour=0, minute=0)
    day_dts = [dt+timedelta(hours=t) for t in range(5)]
    return day_dts

def GOES_doesnt_already_exists(goes_dl_loc, fn_head):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    file_list = glob.glob('{}{}*_{}_*.nc'.format(goes_dl_loc, sat_num, start_scan))
    if len(file_list) > 0:
        print("FILE THAT ALREADY EXIST:", file_list[0], flush=True)
        return False
    return True

def get_dt_str(dt):
    hr = dt.hour
    hr = str(hr).zfill(2)
    tt = dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    yr = dt.year
    return hr, dn, yr


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

def get_scn(fns, to_load, res=2000, reader='abi_l1b'):
    extent = get_extent()
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

def save_reflectances(day_sat_files, dn, season_dict, band, goes_dl_loc):
    for sat_fn in day_sat_files:
        sat_fn = goes_dl_loc + sat_fn.split('/')[-1]
        scn = get_scn([sat_fn], [band]) # get satpy scn object
        band_data = scn[band].compute().data
        band_data[np.isnan(band_data)] = 0
        season_dict[dn].append(band_data.flatten())
    return season_dict

def for_a_day(dt, season_dict, band, sat_num='16'):
    hr, dn, yr = get_dt_str(dt)
    goes_dl_loc = get_goes_dl_loc(yr, dn)
    sat_fns_to_dl = []
    valid_times = get_day_dts(dt)
    if valid_times and sat_num:
        fn_heads, sat_fns = get_sat_files(valid_times, sat_num)
    else:
        sat_fns = None
    if sat_fns:
        for idx, sat_fn_entry in enumerate(sat_fns):
            fn_head = fn_heads[idx]
            if GOES_doesnt_already_exists(goes_dl_loc, fn_head):
                sat_fns_to_dl.extend(sat_fn_entry)
    sat_fns_to_dl = check_goes_exists(sat_fns_to_dl)
    sat_fns_to_dl.sort()
    for sat_fn in sat_fns_to_dl:
        print(sat_fn)
    print(len(sat_fns_to_dl))
    x = input('stop')
    if sat_fns_to_dl:
        p = Pool(8)
        p.map(download_sat_files, sat_fns_to_dl)
    season_dict = save_reflectances(sat_fns_to_dl, dn, season_dict, band, goes_dl_loc)
    return season_dict

def get_season(start_dn, end_dn):
    if start_dn in list(range(335, 366)) + list(range(1, 60)) and end_dn in list(range(335, 366)) + list(range(1, 60)):
        return 'winter'
    if start_dn in list(range(60, 152)) and end_dn in list(range(60, 152)):
        return 'spring'
    if start_dn in list(range(152, 224)) and end_dn in list(range(152, 224)):
        return 'summer'
    if start_dn in list(range(224, 335)) and end_dn in list(range(224, 335)):
        return 'fall'
    else:
        print('MAKE SURE start_dn and end_dn are in same season!!!')
        return None

def get_season_dict(start_dn, end_dn, band):
    season = get_season(int(start_dn), int(end_dn))
    fn = f"./data_pkls/{season}_{band}_data.pkl"
    with open(fn, 'rb') as f:
        season_dict = pickle.load(f)
    return season_dict, fn

def main(start_dn, end_dn, yr):
    band = 'C14'
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    season_dict, season_fn = get_season_dict(start_dn, end_dn, band)
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append({'day_number': dn, 'year': yr})
    dates.reverse()
    for date in dates:
        day_dt = pytz.utc.localize(datetime.strptime('{}{}'.format(date['year'], date['day_number']), '%Y%j')) # convert to datetime object
        season_dict = for_a_day(day_dt, season_dict, band)
        with open(season_fn, 'wb') as f:
            pickle.dump(season_dict, f)
        start = time.time()
        print("Time elapsed for data download for day {}{}: {}s".format(date['year'], date['day_number'], int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
