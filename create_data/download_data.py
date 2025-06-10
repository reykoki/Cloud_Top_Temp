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
import shutil
import wget
from datetime import timedelta
from get_goes import get_sat_files, get_goes_dl_loc, get_file_locations, download_sat_files, check_goes_exists, doesnt_already_exists
#from get_null_df import *

global goes_dir
#goes_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/GOES/'
goes_dir = '/scratch/alpine/mecr8410/Cloud_Top_Temp/GOES/'

# get list of datetimes when the solar zenith angle for far east and west boundaries are less than 50 deg
def get_sun_up_dts(dt,max_sza=70.0, west_lon=-125.0, west_lat=40.0, east_lon=-110.0, east_lat=40.0):
    dt = dt.replace(hour=0, minute=0)
    day_dts = [dt+timedelta(hours=t) for t in range(24)]
    szas_west = np.asarray([astronomy.sun_zenith_angle(day_dt, west_lon, west_lat) for day_dt in day_dts])
    szas_east = np.asarray([astronomy.sun_zenith_angle(day_dt, east_lon, east_lat) for day_dt in day_dts])
    idx_east = list(np.where(szas_east<max_sza)[0])
    idx_west = list(np.where(szas_west<max_sza)[0])
    overlap = list(set(list(idx_east)) & set(list(idx_west)))
    sun_up_dts = [day_dts[idx] for idx in overlap]
    return sun_up_dts

def get_dt_every_n_hrs(dt, n_hrs=3):
    dt = dt.replace(hour=0, minute=0)
    day_dts = [dt+timedelta(hours=t) for t in np.arange(0,24,n_hrs)]
    return day_dts

def get_dt_str(dt):
    hr = dt.hour
    hr = str(hr).zfill(2)
    tt = dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    yr = dt.year
    return hr, dn, yr

def for_a_dt(dt, sat_num='16'):
    hr, dn, yr = get_dt_str(dt)
    goes_dl_loc = get_goes_dl_loc(yr, dn)
    sat_fns_to_dl = []
    valid_times = [dt]
    if valid_times and sat_num:
        fn_heads, sat_fns = get_sat_files(valid_times, sat_num)
    else:
        sat_fns = None
    if sat_fns:
        for idx, sat_fn_entry in enumerate(sat_fns):
            fn_head = fn_heads[idx]
            if doesnt_already_exists(yr, dn, fn_head):
                sat_fns_to_dl.extend(sat_fn_entry)
    sat_fns_to_dl = check_goes_exists(sat_fns_to_dl)
    print(sat_fns_to_dl)
    for sat_fn in sat_fns_to_dl:
        print(sat_fn)
    if sat_fns_to_dl:
        print(sat_fn)
    if sat_fns_to_dl:
        p = Pool(4)
        p.map(download_sat_files, sat_fns_to_dl)

def for_a_day(dt, sat_num='16'):
    hr, dn, yr = get_dt_str(dt)
    goes_dl_loc = get_goes_dl_loc(yr, dn)
    sat_fns_to_dl = []
    #valid_times = get_sun_up_dts(dt)
    valid_times = get_dt_every_n_hrs(dt)
    if valid_times and sat_num:
        fn_heads, sat_fns = get_sat_files(valid_times, sat_num)
    else:
        sat_fns = None
    if sat_fns:
        for idx, sat_fn_entry in enumerate(sat_fns):
            fn_head = fn_heads[idx]
            if doesnt_already_exists(yr, dn, fn_head):
                sat_fns_to_dl.extend(sat_fn_entry)
    sat_fns_to_dl = check_goes_exists(sat_fns_to_dl)
    for sat_fn in sat_fns_to_dl:
        print(sat_fn)
    if sat_fns_to_dl:
        p = Pool(8)
        p.map(download_sat_files, sat_fns_to_dl)

def main(start_dn, end_dn, yr):
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append({'day_number': dn, 'year': yr})
    dates.reverse()
    for date in dates:
        day_dt = pytz.utc.localize(datetime.strptime('{}{}'.format(date['year'], date['day_number']), '%Y%j')) # convert to datetime object
        for_a_day(day_dt)
        start = time.time()
        print("Time elapsed for data download for day {}{}: {}s".format(date['year'], date['day_number'], int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
