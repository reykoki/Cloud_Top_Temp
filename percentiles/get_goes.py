import os
import random
from datetime import datetime
import pytz
from datetime import timedelta
import boto3
from botocore import UNSIGNED
from botocore.client import Config

global client
client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def get_goes_dl_loc(yr, dn):
    goes_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/GOES/'
    global goes_dl_loc
    goes_dl_loc = '{}{}/{}/'.format(goes_dir, yr, dn)
    return goes_dl_loc

def get_file_locations(use_fns):
    file_locs = []
    for file_path in use_fns:
        fn_dl_loc = goes_dl_loc+file_path.split('/')[-1]
        if os.path.exists(fn_dl_loc):
            file_locs.append(fn_dl_loc)
        else:
            print('{} doesnt exist'.format(fn_dl_loc))
    return file_locs

def get_mode(dt):
    M3_to_M6 = pytz.utc.localize(datetime(2019, 4, 1, 0, 0)) # April 2019 switch from Mode 3 to Mode 6 (every 15 to 10 mins)
    if dt < M3_to_M6:
        mode = 'M3'
    else:
        mode = 'M6'
    return mode

def get_goes_west(sat_num, dt):
    G18_op_dt = pytz.utc.localize(datetime(2022, 7, 28, 0, 0))
    G17_op_dt = pytz.utc.localize(datetime(2018, 8, 28, 0, 0))
    if dt >= G18_op_dt:
        sat_num = '18'
    if dt < G17_op_dt:
        print("G17 not launched yet")
        sat_num = '16'
    return sat_num

def get_GOES_file_loc(curr_time, mode, sat_num, band):
    yr = curr_time.year
    dn = curr_time.strftime('%j')
    hr = curr_time.hour
    hr = str(hr).zfill(2)
    band_prefix = 'ABI-L1b-RadF/{}/{}/{}/OR_ABI-L1b-RadF-{}{}_G{}_s{}{}{}'.format(yr, dn, hr, mode, band, sat_num, yr, dn, hr)
    band_filelist = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=band_prefix)
    if band_filelist['KeyCount'] >= 1:
        idx = random.randint(0,band_filelist['KeyCount'])
        band_fn = band_filelist['Contents'][idx-1]['Key']
        return [band_fn]

def get_sat_files(time_list, sat_num, band='C14'):

    all_fn_heads = []
    all_sat_fns = []

    if sat_num == '17':
        sat_num = get_goes_west(sat_num, time_list[0])
    mode = get_mode(time_list[0])

    for curr_time in time_list:
        sat_fns = get_GOES_file_loc(curr_time, mode, sat_num, band)
        print(sat_fns)
        if sat_fns:
            fn_head = sat_fns[0].split('{}_'.format(band))[-1].split('.')[0].split('_e2')[0]
            all_fn_heads.append(fn_head)
            all_sat_fns.append(sat_fns)

    if len(all_sat_fns)>0:
        all_sat_fns = [list(item) for item in set(tuple(row) for row in all_sat_fns)]
        all_fn_heads = list(set(all_fn_heads))
        return all_fn_heads, all_sat_fns
    return None, None

def check_goes_exists(sat_files):
    sat_files = list(set(sat_files))
    sat_fns_to_dl = []
    for sat_file in sat_files:
        fn = sat_file.split('/')[-1]
        fn_dl_loc = goes_dl_loc+fn
        sat_num = fn.split('G')[-1][:2]
        if os.path.exists(fn_dl_loc) is False:
            sat_fns_to_dl.append(sat_file)
    return sat_fns_to_dl

def download_sat_files(sat_file):
    fn = sat_file.split('/')[-1]
    fn_dl_loc = goes_dl_loc+fn
    sat_num = fn.split('G')[-1][:2]
    print('downloading {}'.format(fn))
    try:
        client.download_file(Bucket='noaa-goes{}'.format(sat_num), Key=sat_file, Filename=fn_dl_loc)
    except Exception as e:
        print('boto3 failed')
        print(e)
        pass
    return

