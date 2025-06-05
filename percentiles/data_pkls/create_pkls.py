import pickle
from datetime import datetime, timedelta

# Configuration
year = 2020  # Leap year to include day 366
start_month = 1
end_month = 12  # Inclusive
band = "C14"

# Generate and save monthly Julian day dictionaries
for month in range(start_month, end_month + 1):
    first_day = datetime(year, month, 1)
    next_month = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    # Get Julian day numbers for the month
    month_days = [(first_day + timedelta(days=i)).timetuple().tm_yday
                  for i in range((next_month - first_day).days)]

    # Dictionary with Julian days as keys
    month_dict = {'{0:03d}'.format(d): [] for d in month_days}

    # Save to pickle with unpadded month number
    filename = '{}_{}_data.pkl'.format(band, month)
    with open(filename, "wb") as f:
        pickle.dump(month_dict, f)
print("Pickle files saved for months {} through {}.".format(start_month, end_month))
