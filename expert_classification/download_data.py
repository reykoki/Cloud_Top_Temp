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
from random import randrange
from datetime import datetime
import numpy as np
import time
import pytz
import shutil
import wget
from datetime import timedelta
from get_goes import get_sat_files, get_goes_dl_loc, get_file_locations, download_sat_files, check_goes_exists, goes_doesnt_already_exists
from main import sample_doesnt_exist
#from get_null_df import *

global goes_dir
#goes_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/GOES/'
#goes_dir = '/scratch/alpine/mecr8410/Cloud_Top_Temp/GOES/'
goes_dir = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/Cloud_Top_Temp/expert_classification/GOES/'
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
    return [random.choice(sun_up_dts)]

def get_random_dt(dt, n_hrs=3):
    dt = dt.replace(hour=0, minute=0)
    rand_hr = randrange(24)
    day_dts = [dt+timedelta(hours=rand_hr)]
    return day_dts

def get_dt_str(dt):
    hr = dt.hour
    hr = str(hr).zfill(2)
    tt = dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    yr = dt.year
    return hr, dn, yr

def for_a_day(dt, sat_num='16'):
    hr, dn, yr = get_dt_str(dt)
    goes_dl_loc = get_goes_dl_loc(yr, dn)
    sat_fns_to_dl = []
    valid_times = get_sun_up_dts(dt)
    #valid_times = get_random_dt(dt)

    if valid_times and sat_num:
        fn_heads, sat_fns = get_sat_files(valid_times, sat_num)
    else:
        sat_fns = None
    if sat_fns:
        for idx, sat_fn_entry in enumerate(sat_fns):
            fn_head = fn_heads[idx]
            #if goes_doesnt_already_exists(yr, dn, fn_head) and sample_doesnt_exist(yr, dn, fn_head):
            if goes_doesnt_already_exists(yr, dn, fn_head):
                sat_fns_to_dl.extend(sat_fn_entry)
    sat_fns_to_dl = check_goes_exists(sat_fns_to_dl)
    for sat_fn in sat_fns_to_dl:
        print(sat_fn)
    if sat_fns_to_dl:
        p = Pool(8)
        p.map(download_sat_files, sat_fns_to_dl)

def main():
    yr = 2023
    dns = random.sample(range(1, 365), 50)
    print(dns)
    dns = [325, 83, 278, 169, 129, 248, 180, 200, 154, 236, 33, 277, 251, 344, 282, 310, 156, 205, 29, 324, 14, 132, 226, 161, 176, 349, 153, 128, 290, 337, 9, 260, 140, 299, 18, 204, 196, 183, 166, 203, 160, 104, 362, 228, 164, 173, 199, 346, 353, 186]
    dates = []
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
    main()
