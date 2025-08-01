import pickle
import glob
import ray
import sys
import logging
import geopandas
import os
from datetime import datetime
import numpy as np
import time
import pytz
import shutil
from create_data import create_data_truth
from get_goes import get_goes_dl_loc

global ray_par_dir
ray_par_dir = "/tmp/"
global full_data_dir
#full_data_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Cloud_Top_Temp/cloud_data/'
full_data_dir = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/Cloud_Top_Temp/cloud_data/'

def sat_files_exist(file_locs):
    for sat_fn in file_locs:
        if not os.path.exists(sat_fn):
            return False
    return True

def sample_doesnt_exist(yr, dn, fn_head):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    file_list = glob.glob('{}truth/{}/{}/{}_{}_*.tif'.format(full_data_dir, yr, dn, sat_num, start_scan))
    if len(file_list) > 0:
        print("FILES THAT ALREADY EXIST:", file_list, flush=True)
        #if len(file_list) > 1:
        #    file_list.sort()
        #    fns_to_rm = file_list[1:]
        #    for fn in fns_to_rm:
        #        os.remove(fn)
        #        os.remove(fn.replace('truth','data'))
        return False
    #bad_file_list = glob.glob('{}bad_img/{}_{}_*_{}.tif'.format(full_data_dir, sat_num, start_scan))

    #if len(bad_file_list) > 0:
    #    print("BAD FILES:", bad_file_list, flush=True)
    #    return False
    return True

def get_yr_dn_month_fn_head(fn_head):
    yr_dn = fn_head.split('_s')[-1][0:7]
    yr = yr_dn[0:4]
    dn = yr_dn[4:]
    dt = datetime.strptime(yr_dn, '%Y%j')
    return yr, dn, dt.month

def get_stats_dict(month, sat):
    bands = ['C14', 'C15', 'C16']
    all_band_dict = {}
    for band in bands:

        pkl_loc = "../percentiles/data_pkls/{}/percentile_stats_{}_{}_2024_{}.pkl".format(band, sat, band, str(month).zfill(2))
        print(pkl_loc)
        with open(pkl_loc, 'rb') as f:
            stats_dict = pickle.load(f)
        all_band_dict.update({band : stats_dict})
    return all_band_dict

@ray.remote(max_calls=1)
def iter_sample(sample_dict):
    create_data_truth(sample_dict["files_for_sample"], sample_dict["year"], sample_dict["dn"], sample_dict["fn_head"], sample_dict["band_stats"])
    return

def run_no_ray(sample_list):
    for sample in sample_list:
        iter_sample(sample)
    return

def run_remote(sample_list):
    try:

        ray.get([iter_sample.remote(sample) for sample in sample_list])
    
        return
    except Exception as e:
        print(e)
        return

def get_sample_list(goes_dl_loc, sat):
    sample_list = []

    C14_list = glob.glob('{}/*C14_*{}*.nc'.format(goes_dl_loc, sat))
    C14_list.sort()
    if C14_list:
        fn_head = C14_list[0].split('C14_')[-1].split('.')[0].split('_e2')[0]
        yr, dn, month = get_yr_dn_month_fn_head(fn_head)
        band_stats = get_stats_dict(month, sat)
        for C14_sample in C14_list:
            fn_head = C14_sample.split('C14_')[-1].split('.')[0].split('_e2')[0]
            files_for_sample = glob.glob('{}/*{}*.nc'.format(goes_dl_loc, fn_head))
            files_for_sample.sort()
            doesnt_exist = sample_doesnt_exist(yr, dn, fn_head)
            if len(files_for_sample) == 4 and doesnt_exist:
                fn_head = C14_sample.split('C14_')[-1].split('.')[0].split('_e2')[0]
                sample_dict = {"files_for_sample": files_for_sample, "fn_head": fn_head, "sat": sat, "year": yr, "dn": dn, "band_stats": band_stats}
                sample_list.append(sample_dict)
    return sample_list

def get_dt_str(dt):
    hr = dt.hour
    hr = str(hr).zfill(2)
    tt = dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    yr = dt.year
    return hr, dn, yr

def iter_samples(dt, sat):
    hr, dn, yr = get_dt_str(dt)
    goes_dl_loc = get_goes_dl_loc(yr, dn)
    sample_list = get_sample_list(goes_dl_loc, sat)
    if sample_list:
        ray_dir = "{}{}{}".format(ray_par_dir,yr,dn)
        if not os.path.isdir(ray_dir):
            os.mkdir(ray_dir)
        ray.init(num_cpus=8, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1', object_store_memory=10**9)
        run_remote(sample_list)
        #run_no_ray(sample_list)
        ray.shutdown()
        shutil.rmtree(ray_dir)

def main(start_dn, end_dn, yr, sat):
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append({'day_number': dn, 'year': yr})
    dates.reverse()
    for date in dates:
        start = time.time()
        print(date)
        day_dt = pytz.utc.localize(datetime.strptime('{}{}'.format(date['year'], date['day_number']), '%Y%j')) # convert to datetime object
        iter_samples(day_dt, sat)
        print("Time elapsed for data creation for day {}{}: {}s".format(date['year'], date['day_number'], int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    sat = sys.argv[4]
    #sat = "G16" 
    main(start_dn, end_dn, yr, sat)
