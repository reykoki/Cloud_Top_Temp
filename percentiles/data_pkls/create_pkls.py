import pickle

season_ranges = {
        'winter': list(range(335, 366)) + list(range(1, 60)),   # Dec 1 – Feb 28
        'spring': list(range(60, 152)),                         # Mar 1 – May 31
        'summer': list(range(152, 244)),                        # Jun 1 – Aug 31
        'fall': list(range(244, 335)),                          # Sep 1 – Nov 30
}

for season, doy_list in season_ranges.items():
    season_dict = {}
    for doy in doy_list:
        doy_str = f"{doy:03d}"
        season_dict[doy_str] = []

    # Save the season dictionary to a separate pickle file
    filename = f"{season}_C14_data.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(season_dict, f)
    print(f"Pickle file created: {filename}")
