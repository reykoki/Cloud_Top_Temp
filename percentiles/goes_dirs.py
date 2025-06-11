import os
import glob


root_dir = './reflectance/'
root_dir = '/scratch/alpine/mecr8410/Cloud_Top_Temp/GOES/'

yrs = [ '2018/', '2019/', '2020/', '2021/', '2022/', '2023/', '2024/']
dns = [str(x).zfill(3) for x in range(1,367)]

for yr in yrs:
    for dn in dns:
        dn_path = root_dir + yr + dn
        if not os.path.exists(dn_path):
            os.makedirs(dn_path)
