import os
import glob


yrs = [ '2018/', '2019/', '2020/', '2021/', '2022/', '2023/', '2024/']
dns = [str(x).zfill(3) for x in range(1,367)]
bands = ['C'+str(x).zfill(2) for x in range(1,17)]
print(bands)
sats = ['G16', 'G18']


def make_stats_dir():
    root_dir = './stats/'
    for yr in yrs:
        for sat in sats:
            for band in bands:
                dn_path = root_dir + yr + '/' + sat + '/' +  band
                if not os.path.exists(dn_path):
                    os.makedirs(dn_path)


def make_reflectance_dir():
    root_dir = './reflectance/'
    yr = '2023'
    for sat in sats:
        for band in bands:
            for dn in dns:
                dn_path = root_dir + yr + '/' + sat + '/' +  band + '/' +  dn
                if not os.path.exists(dn_path):
                    os.makedirs(dn_path)

def make_goes_dir():
    root_dir = '/scratch/alpine/mecr8410/Cloud_Top_Temp/GOES/'
    for yr in yrs:
        for dn in dns:
            dn_path = root_dir + yr + dn
            if not os.path.exists(dn_path):
                os.makedirs(dn_path)


make_stats_dir()
#make_reflectance_dir()
