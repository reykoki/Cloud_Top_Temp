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
from grab_smoke import get_smoke_fn_url
from create_data import create_data_truth
from pseudo import find_best_data
from get_goes import get_goes_dl_loc

global smoke_dir
smoke_dir = "/scratch1/RDARCH/rda-ghpcs/Rey.Koki/smoke/"
global ray_par_dir
ray_par_dir = "/tmp/"
global pseudo_dir
pseudo_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL_SmokeViz/'
global full_data_dir
full_data_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/full_dataset/'

def sat_files_exist(file_locs):
    for sat_fn in file_locs:
        if not os.path.exists(sat_fn):
            return False
    return True

def sample_doesnt_exist(yr, dn, fn_head, idx, density):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    file_list = glob.glob('{}truth/{}/{}/{}/{}_{}_*_{}.tif'.format(full_data_dir, yr, density, dn, sat_num, start_scan, idx))
    if len(file_list) > 0:
        print("FILES THAT ALREADY EXIST:", file_list, flush=True)
        if len(file_list) > 1:
            file_list.sort()
            fns_to_rm = file_list[1:]
            for fn in fns_to_rm:
                os.remove(fn)
                os.remove(fn.replace('truth','data'))
        return False
    bad_file_list = glob.glob('{}bad_img/{}_{}_*_{}.tif'.format(full_data_dir, sat_num, start_scan, idx))

    if len(bad_file_list) > 0:
        print("BAD FILES:", bad_file_list, flush=True)
        return False
    return True

def check_smoke_rows(smoke_rows):
    checked_smoke_rows = []
    for smoke_row in smoke_rows:
        idx = smoke_row['idx']
        density = smoke_row['density']
        yr = smoke_row['yr']
        dn = smoke_row['dn']
        file_locs = smoke_row['sat_file_locs']
        fn_head = file_locs[0].split('C01_')[-1].split('.')[0].split('_e2')[0]
        no_sample = sample_doesnt_exist(yr, dn, fn_head, idx, density)
        if len(file_locs) == 3 and sat_files_exist(file_locs) and no_sample:
            checked_smoke_rows.append(smoke_row)
    return checked_smoke_rows

def get_yr_dn_fn_head(fn_head):
    yr_dn = fn_head.split('_s')[-1][0:7]
    yr = yr_dn[0:4]
    dn = yr_dn[4:]
    return yr, dn

@ray.remote(max_calls=1)
def iter_sample(files_for_sample):
    fn_head = files_for_sample.split('C14_')[-1].split('.')[0].split('_e2')[0]
    yr, dn = get_yr_dn_fn_head(fn_head)
    create_data_truth(files_for_sample, yr, dn, fn_head)
    return

def run_no_ray(smoke_rows):
    for smoke_row in smoke_rows:
        iter_rows(smoke_row)
    return

def run_remote(sample_list):
    try:
        ray.get([iter_sample.remote(sample) for sample in sample_list])
        return
    except Exception as e:
        print(e)
        return

def get_sample_list(goes_dl_loc):
    file_list = []
    C14_list = glob.glob('{}/*{}*.nc'.format(goes_dl_loc, fn_head))
    for C14_sample in C14_list:
        fn_head = C14_sample.split('C14_')[-1].split('.')[0].split('_e2')[0]
        files_for_sample = glob.glob('{}/*{}*.nc'.format(goes_dl_loc, fn_head))
        if len(file_for_sample) == 4:
            file_list.append(file_for_sample)
    return file_list

def iter_samples(dt):
    hr, dn, yr = get_dt_str(dt)
    goes_dl_loc = get_goes_dl_loc(yr, dn)
    sample_list = get_sample_list(goes_dl_loc)
    ray_dir = "{}{}{}".format(ray_par_dir,yr,dn)
    if not os.path.isdir(ray_dir):
        os.mkdir(ray_dir)
    ray.init(num_cpus=3, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1', object_store_memory=10**9)
    run_remote(checked_smoke_rows)
    #run_no_ray(checked_smoke_rows)
    ray.shutdown()
    shutil.rmtree(ray_dir)
    #find_best_data(yr, dn)
                #goes_files = glob.glob('{}*.nc'.format(goes_dl_loc))
                #for goes_fn in goes_files:
                #    os.remove(goes_fn)
        except Exception as e:
            print(e)
            return
def main(start_dn, end_dn, yr):
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append({'day_number': dn, 'year': yr})
    dates.reverse()
    for date in dates:
        start = time.time()
        print(date)
        iter_smoke(date)
        day_dt = pytz.utc.localize(datetime.strptime('{}{}'.format(date['year'], date['day_number']), '%Y%j')) # convert to datetime object
        print("Time elapsed for data creation for day {}{}: {}s".format(date[1], date[0], int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
