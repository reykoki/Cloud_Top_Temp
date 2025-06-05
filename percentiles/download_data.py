import shutil
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
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
import skimage
import shutil
import wget
from datetime import timedelta
from get_goes import get_sat_files, get_goes_dl_loc, get_file_locations, download_sat_files, check_goes_exists
import xarray as xr

global goes_dir
goes_dir = './GOES/'

# get list of datetimes (24 dts per day) chosen random selection per hour
def get_day_dts(dt):
    dt = dt.replace(hour=0, minute=0)
    day_dts = [dt+timedelta(hours=t) for t in range(24)]
    return day_dts

def GOES_doesnt_already_exists(goes_dl_loc, fn_head):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    file_list = glob.glob('{}{}*_{}_*.nc'.format(goes_dl_loc, sat_num, start_scan))
    if len(file_list) > 0:
        print("FILE THAT ALREADY EXIST:", file_list[0])
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

def save_reflectances(day_sat_files, dn, month_dict, band, goes_dl_loc):
    for sat_fn in day_sat_files:
        sat_fn = goes_dl_loc + sat_fn.split('/')[-1]
        if os.path.exists(sat_fn):
            scn = get_scn([sat_fn], [band]) # get satpy scn object
            band_data = scn[band].compute().data
            band_data[np.isnan(band_data)] = 0
            month_dict[dn].append(band_data.flatten())
            os.remove(sat_fn)
    return month_dict

def for_a_day(dt, month_dict, band, sat_num='16'):
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
    if sat_fns_to_dl:
        p = Pool(8)
        p.map(download_sat_files, sat_fns_to_dl)
    month_dict = save_reflectances(sat_fns_to_dl, dn, month_dict, band, goes_dl_loc)
    return month_dict

def get_month_dns(month, year):
    month = int(month)
    year = int(year)
    first_day = datetime(year, month, 1)
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    # Get all Julian day numbers for that month
    month_days = [(first_day + timedelta(days=i)).timetuple().tm_yday
                  for i in range((next_month - first_day).days)]
    month_days.sort()
    return month_days[0], month_days[-1]

def get_month_dict(month, band):
    fn = "./data_pkls/{}_{}_data.pkl".format(band, month)
    with open(fn, 'rb') as f:
        month_dict = pickle.load(f)

    return month_dict, fn

def main(month_number, yr, band):
    dates = []
    start_dn, end_dn = get_month_dns(month_number, yr)
    month_dict, month_fn = get_month_dict(month_number, band)
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append({'day_number': dn, 'year': yr})
    dates.reverse()
    for date in dates:
        day_dt = pytz.utc.localize(datetime.strptime('{}{}'.format(date['year'], date['day_number']), '%Y%j')) # convert to datetime object
        if len(month_dict[date['day_number']]) == 0:
            month_dict = for_a_day(day_dt, month_dict, band)
            with open(month_fn, 'wb') as f:
                pickle.dump(month_dict, f)
        start = time.time()
        print("Time elapsed for data download for day {}{}: {}s".format(date['year'], date['day_number'], int(time.time() - start)))

if __name__ == '__main__':
    month_number = sys.argv[1]
    yr = sys.argv[2]
    band = sys.argv[3] #C14
    main(month_number, yr, band)
