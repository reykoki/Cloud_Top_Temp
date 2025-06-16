import numpy as np
import sys
import pickle
from pathlib import Path
from calendar import monthrange
from datetime import datetime, timedelta

def get_days_for_month_window(year, month):
    prev_month = (month - 2) % 12 + 1
    next_month = month % 12 + 1
    months = [prev_month, month, next_month]
    days = []
    for m in months:
        y = year + (m < month - 1) - (m > month + 1)
        for d in range(1, monthrange(year, m)[1] + 1):
            dt = datetime(year, m, d)
            days.append((dt - datetime(year, 1, 1)).days + 1)
    return sorted(set(days))

def compute_monthly_stats(year, month, base_path, band, percentiles, sat):
    print("Processing month {:02d}".format(month))
    days = get_days_for_month_window(year, month)
    print(days)
    all_vals = []
    for doy in days:
        npy_path = Path(base_path) / "{:03d}/{}_{}.npy".format(doy, sat, band)
        if npy_path.exists():
            try:
                arr = np.load(npy_path, mmap_mode='r')
                all_vals.append(arr)
            except Exception as e:
                print("Error reading {}: {}".format(npy_path, e))
    if not all_vals:
        print("No data for month {:02d}".format(month))
        return None

    flat_vals = np.concatenate(all_vals)
    print(len(flat_vals))
    print(len(flat_vals)*4/(1024**3), 'GB')
    pct_vals = np.percentile(flat_vals, percentiles)
    stats = {
        'percentiles': dict(zip(percentiles, pct_vals)),
        'mean': float(np.mean(flat_vals)),
        'std': float(np.std(flat_vals)),
        'median': float(np.median(flat_vals))
    }
    return stats


def main(month_number, year, band, sat):
    base_path = './reflectance/{}/{}/{}/'.format(year, sat, band)
    percentiles = [
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        25.0, 50.0, 75.0,
        90.0, 91.0, 92.0, 93.0, 94.0, 95.0,
        95.5, 96.0, 96.5, 97.0, 97.5, 98.0, 98.5, 99.0, 99.5
    ]
    if month_number == 'all':
        months = list(range(1,13))
    else:
        months = [int(month_number)]
    for month in months:
        stats = compute_monthly_stats(int(year), month, base_path, band, percentiles, sat)
        if stats is not None:
            output_file = './data_pkls/{}/percentile_stats_{}_{}_{}_{}.pkl'.format(band, sat, band, year, str(month).zfill(2))
            with open(output_file, 'wb') as f:
                pickle.dump(stats, f)
            print("Saved to {}".format(output_file))

if __name__ == '__main__':
    sat= sys.argv[1] #C14
    band = sys.argv[2] #C14
    yr = sys.argv[3]
    month_number = sys.argv[4]
    main(month_number, yr, band, sat)
