import pickle
import calendar

daynum = 1
for month in range(1, 13):
    days_in_month = calendar.monthrange(2020, month)[1]  # 2020 = leap year
    # Create dict with keys as 3-digit day numbers and empty list values
    month_dict = {f'{daynum + i:03d}': [] for i in range(days_in_month)}

    # Save dict to pickle
    with open(f'{month:02d}.pkl', 'wb') as f:
        pickle.dump(month_dict, f)

    daynum += days_in_month
