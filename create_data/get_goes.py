import os
import glob
from datetime import datetime
import pytz
from datetime import timedelta
import boto3
from botocore import UNSIGNED
from botocore.client import Config

global client
client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

global goes_dir
#goes_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/GOES/'
goes_dir = '/scratch/alpine/mecr8410/Cloud_Top_Temp/GOES/'

def doesnt_already_exists(yr, dn, fn_head):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    file_list = glob.glob('{}{}/{}/{}*{}*.nc'.format(goes_dir, yr, dn, sat_num, start_scan))
    print(file_list)
    if len(file_list) > 0:
        print("FILE THAT ALREADY EXIST:", file_list[0], flush=True)
        return False
    return True


def get_goes_dl_loc(yr, dn):
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

def get_closest_file(fns, best_time, sat_num):
    diff = timedelta(days=100)
    use_fns = []
    for fn in fns:
        starts = []
        if 'C14' in fn:
            s_e = fn.split('_')[3:5]
            start = s_e[0]
            end = s_e[1][0:11]
            C15_fn = 'C15_G{}_{}_{}'.format(sat_num, start, end)
            C16_fn = 'C16_G{}_{}_{}'.format(sat_num, start, end)
            for f in fns:
                if C15_fn in f:
                   C15_fn = f
                elif C16_fn in f:
                   C16_fn = f
            if 'nc' in C15_fn and 'nc' in C16_fn:
                start = s_e[0][1:-3]
                s_dt = pytz.utc.localize(datetime.strptime(start, '%Y%j%H%M'))
                if diff > abs(s_dt - best_time):
                    diff = abs(s_dt - best_time)
                    use_fns = [fn, C15_fn, C16_fn]
    return use_fns

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

def diagnose_filelist(curr_time, mode, sat_num, yr, dn, hr, mn):
    print('need diagnosis')

    diff = timedelta(minutes=10)
    #C14_prefix = 'ABI-L1b-RadC/{}/{}/{}/OR_ABI-L1b-RadC-{}C14_G{}_s{}{}{}'.format(yr, dn, hr, mode, sat_num, yr, dn, hr)
    #print(C14_prefix)
    #C14_filelist = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C14_prefix)
    #print(C14_filelist)
    use_entry = None
    #if C14_filelist == 0:
    if mode == 'M3':
        mode2 = 'M6'
    else:
        mode2 = 'M3'
    C14_prefix = 'ABI-L1b-RadF/{}/{}/{}/OR_ABI-L1b-RadF-{}C14_G{}_s{}{}{}'.format(yr, dn, hr, mode2, sat_num, yr, dn, hr, mn)
    C14_filelist = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C14_prefix)
    G17_end_dt = pytz.utc.localize(datetime(2023, 1, 10, 0, 0))
    if C14_filelist['KeyCount'] == 0 and sat_num == '18' and curr_time < G17_end_dt:
        sat_num = '17'
        C14_prefix = 'ABI-L1b-RadF/{}/{}/{}/OR_ABI-L1b-RadF-{}C14_G{}_s{}{}{}'.format(yr, dn, hr, mode2, sat_num, yr, dn, hr)
        C14_filelist = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C14_prefix)

    if C14_filelist['KeyCount'] == 0:
        return None, None
    for entry in C14_filelist['Contents']:
        start = entry['Key'].split('_')[3:5][0][1:-3]
        s_dt = pytz.utc.localize(datetime.strptime(start, '%Y%j%H%M'))
        if diff > abs(s_dt - curr_time):
            diff = abs(s_dt - curr_time)
            use_entry = entry['Key']
    return use_entry, C14_prefix

def get_GOES_file_loc(curr_time, mode, sat_num):
    yr = curr_time.year
    dn = curr_time.strftime('%j')
    hr = curr_time.hour
    hr = str(hr).zfill(2)
    mn = curr_time.minute
    mn = str(mn).zfill(2)
    C14_prefix = 'ABI-L1b-RadF/{}/{}/{}/OR_ABI-L1b-RadF-{}C14_G{}_s{}{}{}{}'.format(yr, dn, hr, mode, sat_num, yr, dn, hr, mn)
    C14_filelist = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C14_prefix)
    if C14_filelist['KeyCount'] != 1:
        C14_fn, C14_prefix = diagnose_filelist(curr_time, mode, sat_num, yr, dn, hr, mn)
    else:
        C14_fn = C14_filelist['Contents'][0]['Key']
    if C14_fn:
        C15_prefix = C14_prefix.replace('C14', 'C15')
        C16_prefix = C14_prefix.replace('C14', 'C16')
        CTT_prefix = 'ABI-L2-ACHTF/{}/{}/{}/OR_ABI-L2-ACHTF-{}_G{}_s{}{}{}{}'.format(yr, dn, hr, mode, sat_num, yr, dn, hr, mn)
        try:
            C15_fn = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C15_prefix)['Contents'][0]['Key']
            C16_fn = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C16_prefix)['Contents'][0]['Key']
            CTT_fn = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=CTT_prefix)['Contents'][0]['Key']
            return [C14_fn, C15_fn, C16_fn, CTT_fn]
        except Exception as e:
            print('could not find accomanying files for: {}'.format(C14_fn))
            print(e)
            return []

def get_sat_files(time_list, sat_num):

    all_fn_heads = []
    all_sat_fns = []

    if sat_num == '17':
        sat_num = get_goes_west(sat_num, time_list[0])
    mode = get_mode(time_list[0])

    for curr_time in time_list:
        sat_fns = get_GOES_file_loc(curr_time, mode, sat_num)
        print(sat_fns)
        if sat_fns:
            fn_head = sat_fns[0].split('C14_')[-1].split('.')[0].split('_e2')[0]
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

