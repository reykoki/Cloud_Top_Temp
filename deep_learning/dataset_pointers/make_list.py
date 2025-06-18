import os
import pickle
import glob
import random

#yrs = ['2018', '2019', '2020', '2021', '2024']
yrs = ['2024']
val_yr = ['2023']
test_yr = ['2023']

truth_dir = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/Cloud_Top_Temp/cloud_data/truth/'


def get_train_fns(yrs):
    truth_fns = []
    data_fns = []
    for yr in yrs:
        yr_truth_dir = truth_dir + yr + '/'
    print(yr_truth_dir)
    yr_truth_fns = glob.glob('{}/*/*.tif'.format(yr_truth_dir))
    yr_data_fns = [s.replace('truth','data') for s in yr_truth_fns]
    truth_fns.extend(yr_truth_fns)
    data_fns.extend(yr_data_fns)
    return truth_fns, data_fns

train_truth, train_data_fns = get_train_fns(yrs)
val_truth, val_data_fns = get_train_fns(val_yr)
test_truth, test_data_fns = get_train_fns(test_yr)

data_dict = {'train': {'truth': train_truth, 'data': train_data_fns},
             'val': {'truth': val_truth, 'data': val_data_fns},
             'test': {'truth': test_truth, 'data': test_data_fns}}

print('number of train samples:', len(train_truth))
print('number of val samples:', len(val_truth))
print('number of test samples:', len(test_truth))
print(len(train_truth)+len(val_truth)+len(test_truth))

with open('cloud.pkl', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
